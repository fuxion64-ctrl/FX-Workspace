"""Audit regressions: isolated stores and fake downloads, no real model or network."""
import base64
import hashlib
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
import zipfile

import chat_store
import composer_tools
import fs_tools
import httpx
import llama_local
import model_finder
import model_scan
import stream_hub


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_attachment_named_metadata_round_trips_and_adopts(self):
        original = b'{"user": "document"}'
        source = composer_tools.FileStore(str(self.root / 'source'))
        dest = composer_tools.FileStore(str(self.root / 'dest'))
        for name in ('meta.json', 'META.JSON'):
            meta = source.save(name, base64.b64encode(original).decode())
            self.assertEqual(Path(source.get(meta['id'])['path']).read_bytes(), original)
        mapping = dest.adopt(source)
        for fid in mapping.values():
            self.assertEqual(Path(dest.get(fid)['path']).read_bytes(), original)

    def test_chat_flag_update_cannot_overwrite_concurrent_message_save(self):
        store = chat_store.ChatStore(str(self.root / 'chats'))
        cid = store.new_id()
        store.save({'id': cid, 'transcript': [{'r': 'u', 't': 'old'}]})
        loaded, release, writer_started = threading.Event(), threading.Event(), threading.Event()
        original_load = store.load
        errors = []
        def slow_load(chat_id):
            result = original_load(chat_id)
            if threading.current_thread().name == 'flag-editor':
                loaded.set()
                release.wait(2)
            return result
        def flags():
            try:
                store.set_flags(cid, pinned=True)
            except Exception as e:
                errors.append(e)
        def message():
            writer_started.set()
            try:
                store.save({'id': cid, 'transcript': [{'r': 'u', 't': 'new'}]})
            except Exception as e:
                errors.append(e)
        with patch.object(store, 'load', side_effect=slow_load):
            editor = threading.Thread(target=flags, name='flag-editor')
            editor.start()
            self.assertTrue(loaded.wait(2))
            writer = threading.Thread(target=message)
            writer.start()
            self.assertTrue(writer_started.wait(2))
            writer.join(.1)
            release.set()
            editor.join(3)
            writer.join(3)
        self.assertFalse(errors)
        result = store.load(cid)
        self.assertEqual(result['transcript'][0]['t'], 'new')
        self.assertTrue(result['pinned'])

    def test_zip_cannot_write_through_existing_symlink(self):
        dest, outside = self.root / 'dest', self.root / 'outside'
        dest.mkdir()
        outside.mkdir()
        (outside / 'secret.txt').write_text('original')
        try:
            (dest / 'link').symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest('Symlinks unavailable')
        archive = self.root / 'input.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('link/secret.txt', 'overwritten')
        result = fs_tools.unzip_file(str(archive), str(dest))
        self.assertFalse(result['ok'])
        self.assertEqual((outside / 'secret.txt').read_text(), 'original')

    def test_existing_model_is_verified_even_when_size_matches(self):
        dest = self.root / 'model.gguf'
        dest.write_bytes(b'evil')
        part = {'path': 'model.gguf', 'size': 4, 'sha256': hashlib.sha256(b'good').hexdigest()}
        model_finder._cancel.clear()
        with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, content=b'good'))) as client:
            model_finder._fetch(client, 'owner/model', part, str(dest))
        self.assertEqual(dest.read_bytes(), b'good')

    def test_partial_response_must_start_at_resume_offset(self):
        dest = self.root / 'model.gguf'
        Path(str(dest) + '.part').write_bytes(b'ab')
        part = {'path': 'model.gguf', 'size': 4, 'sha256': ''}
        model_finder._cancel.clear()
        with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(
                206, headers={'Content-Range': 'bytes 0-1/4'}, content=b'ab'))) as client:
            with self.assertRaises(RuntimeError):
                model_finder._fetch(client, 'owner/model', part, str(dest))
        self.assertFalse(dest.exists())
        self.assertEqual(Path(str(dest) + '.part').read_bytes(), b'ab')

    def test_failed_model_config_replace_preserves_original(self):
        ini = self.root / 'models.ini'
        original = '[*]\nctx-size = 8192\n[M]\nmodel = model.gguf\n'
        ini.write_text(original)
        with patch.dict(os.environ, MODELS_INI=str(ini)), patch('atomic_io.replace', side_effect=OSError('disk failure')):
            with self.assertRaises(OSError):
                llama_local.update_model_config('M', ctx_size=4096)
        self.assertEqual(ini.read_text(), original)

    def test_stream_merge_keeps_chat_privacy_boundaries(self):
        events = [{'kind': 'delta', 'seq': i, 'call': 'same', 'channel': 'content', 'text': text, 'temp': temp}
                  for i, text, temp in [(1, 'private', 'tmp-a'), (2, 'public', None)]]
        self.assertEqual(len(stream_hub.merge(events)), 2)

class ModelConcurrencyTests(unittest.TestCase):
    def test_simultaneous_adds_do_not_duplicate_sections(self):
        with tempfile.TemporaryDirectory() as root:
            ini = Path(root, 'models.ini')
            ini.write_text('[*]\nctx-size = 8192\n')
            model = Path(root, 'model.gguf')
            model.write_bytes(b'GGUF')
            entered, release = threading.Event(), threading.Event()
            outcomes = []
            def options(path):
                if threading.current_thread().name == 'first-add':
                    entered.set()
                    release.wait(2)
                return ''
            def add():
                try:
                    model_scan.add_file(str(model))
                    outcomes.append('added')
                except ValueError:
                    outcomes.append('duplicate')
            with patch.dict(os.environ, MODELS_INI=str(ini)), patch.object(model_scan, '_own_options', side_effect=options):
                first = threading.Thread(target=add, name='first-add')
                second = threading.Thread(target=add, name='second-add')
                first.start()
                self.assertTrue(entered.wait(2))
                second.start()
                second.join(.1)
                release.set()
                first.join(3)
                second.join(3)
                self.assertCountEqual(outcomes, ['added', 'duplicate'])
                self.assertEqual(len(llama_local.list_model_configs()), 1)

    def test_distinct_owners_and_subfolders_get_distinct_download_folders(self):
        folders = []
        def listing(repo):
            return {'files': [{'path': 'a/model.gguf', 'fit': 'gpu'}, {'path': 'b/model.gguf', 'fit': 'gpu'}], 'mmproj': []}
        tree = [{'type': 'file', 'path': p, 'size': 4} for p in ('a/model.gguf', 'b/model.gguf')]
        class Worker:
            def __init__(self, *, args, **kwargs):
                folders.append(args[1])
            def start(self):
                pass
        with patch.object(model_finder, 'files', side_effect=listing), patch.object(model_finder, '_get', return_value=tree), \
                patch.object(model_finder.threading, 'Thread', Worker), patch.dict(model_finder._state, state='idle'):
            for repo, path in [('one/model', 'a/model.gguf'), ('two/model', 'a/model.gguf'), ('one/model', 'b/model.gguf')]:
                model_finder._state['state'] = 'idle'
                self.assertTrue(model_finder.download(repo, path)['ok'])
        self.assertEqual(len(set(folders)), 3)


class DownloadAndStreamTests(unittest.TestCase):
    def test_valid_resume_and_ignored_range(self):
        for code, headers, response in [(206, {'Content-Range': 'bytes 2-3/4'}, b'cd'), (200, {}, b'abcd')]:
            with self.subTest(code=code), tempfile.TemporaryDirectory() as root:
                dest = Path(root, 'model.gguf')
                Path(str(dest) + '.part').write_bytes(b'ab')
                model_finder._cancel.clear()
                with patch.dict(model_finder._state, done=0), httpx.Client(transport=httpx.MockTransport(
                        lambda req: httpx.Response(code, headers=headers, content=response))) as client:
                    model_finder._fetch(client, 'o/m', {'path': 'model.gguf', 'size': 4,
                        'sha256': hashlib.sha256(b'abcd').hexdigest()}, str(dest))
                    self.assertEqual(model_finder._state['done'], 4)
                self.assertEqual(dest.read_bytes(), b'abcd')

    def test_cancelled_download_does_not_reuse_existing_file(self):
        with tempfile.TemporaryDirectory() as root:
            dest = Path(root, 'model.gguf')
            dest.write_bytes(b'abcd')
            model_finder._cancel.set()
            try:
                with self.assertRaises(InterruptedError):
                    model_finder._fetch(Mock(), 'o/m', {'size': 4}, str(dest))
            finally:
                model_finder._cancel.clear()

    def test_bounded_stream_replay_and_forget(self):
        import privacy
        with patch.object(stream_hub, 'MAX_EVENTS', 3):
            hub = stream_hub.StreamHub()
            hub.publish('start', call='x')
            with privacy.scope('tmp-private'):
                hub.publish('delta', call='x', text='private')
            hub.publish('delta', call='x', text='a')
            hub.publish('delta', call='x', text='b')
            self.assertEqual(len(hub._events), 3)
            self.assertEqual(hub.forget('tmp-private'), 1)
            events, seq = hub.since(2, timeout=0)
            self.assertEqual((events[0]['text'], seq), ('ab', 4))
            self.assertEqual(hub.since(seq, timeout=0), ([], seq))


class HistoryPersistenceTests(unittest.TestCase):
    def test_saved_tail_never_starts_with_an_orphan_tool_result(self):
        messages = [{'role': 'user', 'content': 'old'}]
        messages += [{'role': 'assistant', 'tool_calls': [{'id': 'c'}]},
                     {'role': 'tool', 'tool_call_id': 'c', 'content': 'result'},
                     {'role': 'assistant', 'content': 'done'},
                     {'role': 'user', 'content': 'new'},
                     {'role': 'assistant', 'content': 'answer'}]
        # 1.4.8 (audit 2.8): nothing is cut on save any more, so no saved list can start mid-turn - all of it is kept.
        with tempfile.TemporaryDirectory() as root:
            store = chat_store.ChatStore(root)
            cid = store.new_id()
            store.save({'id': cid, 'conversation': messages})
            self.assertEqual(store.load(cid)['conversation'], messages)

    def test_one_large_turn_is_kept_whole_including_synthetic_observations(self):
        messages = [{'role': 'user', 'content': 'task'},
                    {'role': 'assistant', 'tool_calls': [{'id': 'c'}]},
                    {'role': 'user', 'content': '[OBSERVATION] current screen'},
                    {'role': 'tool', 'tool_call_id': 'c', 'content': 'result'},
                    {'role': 'assistant', 'content': 'answer'}]
        with tempfile.TemporaryDirectory() as root, patch.object(chat_store, 'MAX_CONVERSATION', 3):
            store = chat_store.ChatStore(root)
            cid = store.new_id()
            store.save({'id': cid, 'conversation': messages})
            self.assertEqual(store.load(cid)['conversation'], messages)


if __name__ == '__main__':
    unittest.main()
