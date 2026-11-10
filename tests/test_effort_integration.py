"""Saved slider selections reach the real task pipeline; no live models/data writes."""
from __future__ import annotations

import copy
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import core.config as config
import core.state as shared
import effort
import replay_harness as replay
import agent.models as models


def _agent_step(request: dict) -> bool:
    """A streamed reply step of the agent (it offers tools), not a planner / reviewer helper call."""
    return bool(request.get('streamed') and (request.get('tools') or any(
        'You can call these tools' in str(m.get('content', '')) for m in request.get('messages') or [])))


class SliderRuntimeTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        import api.server as server

        self.server = server
        before = dict(shared.app_state)
        self.addCleanup(lambda: (shared.app_state.clear(), shared.app_state.update(before)))
        self.tmp = self.enterContext(tempfile.TemporaryDirectory(prefix='effort-slider-'))
        self.path = pathlib.Path(self.tmp) / 'settings.json'
        self.enterContext(mock.patch.object(config, 'SETTINGS_PATH', str(self.path)))
        self.enterContext(mock.patch.object(server.event_log, 'log_action'))
        self.enterContext(mock.patch.object(server.llm, '_current_model', return_value='scripted-model'))
        self.enterContext(mock.patch.object(server.model_caps, 'get', return_value={'context': 65536, 'tool_protocol': 'native'}))
        shared.app_state.update(settings=copy.deepcopy(config.DEFAULT_SETTINGS), mode=config.MODE_LOCAL,
                                active_project=None, effort_session=None, prep_session=None, run_budget=None,
                                raw_mode=False, turn_voice=False, turn_tools=set(), turn_instructions='',
                                turn_persona='', design_mode=False, events=[], event_seq=0,
                                audit_log=[], audit_seq=0)
        shared.app_state['settings'].update(effort_auto_escalate=False, prompt_prep='off', max_revisions='auto')
        self.assertTrue(config.save_settings(shared.app_state['settings']))
        # Deliberately avoid TestClient's lifespan: no app services start.
        self.client = TestClient(server.app, base_url='http://127.0.0.1:8000')
        self.addCleanup(self.client.close)

    def select(self, level, session=False):
        endpoint = '/api/effort/session' if session else '/api/settings'
        payload = {'level': level} if session else {
            'effort': level, 'for_mode': 'local', 'for_model': 'scripted-model'}
        response = self.client.post(endpoint, json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        selected = self.client.get('/api/effort?mode=local&model=scripted-model').json()
        self.assertEqual((selected['requested'], selected['applied']), (level, level))
        return selected

    def fixture(self, settings):
        fx = replay.load_fixture('effort_change_midrun')
        fx['meta']['model'] = 'scripted-model'
        fx['setup'].pop('settings_at_response', None)
        fx['setup'].update(settings=copy.deepcopy(settings), local_context=65536,
                           model_caps={'context': 65536, 'tool_protocol': 'native'},
                           app_state={'active_project': None, 'effort_session': shared.app_state.get('effort_session')})
        fx['aux_responses'] = [
            {'content': json.dumps({'steps': ['Inspect the requested folders'], 'requirements': ['List text files'],
                                    'success_criteria': ['Requested folders inspected']})},
            {'content': json.dumps({'verdict': 'complete', 'findings': []})}]
        return fx

    @staticmethod
    def started(run):
        return next(e for e in run.events if e.get('phase') == 'run_started')

    def test_each_saved_slider_stop_changes_real_limits_and_local_model_payload(self):
        seen = {}
        for level in effort.LEVELS:
            with self.subTest(level=level):
                selected = self.select(level)
                saved = config.load_settings()
                self.assertEqual(saved['effort_by_model']['scripted-model'], level)
                run = replay.run_fixture(self.fixture(saved))
                self.assertEqual(run.status, 'success', run.message)
                started = self.started(run)
                self.assertEqual(started['effort'], level)
                self.assertEqual(started['limits'], effort.PROFILES[level].limits())
                self.assertEqual(started['limits']['orchestration'], 'none')
                self.assertFalse(started['limits']['delegation'])
                self.assertEqual(next(x['limits'] for x in selected['levels'] if x['id'] == level), started['limits'])
                calls = [r for r in run.requests if _agent_step(r)]
                self.assertTrue(calls)
                profile = effort.PROFILES[level]
                for call in calls:
                    self.assertEqual(call['chat_template_kwargs']['enable_thinking'], profile.thinking)
                    self.assertEqual(call['max_tokens'], config.LOCAL_MAX_TOKENS + (profile.reasoning_tokens if profile.thinking else 0))
                # 1.4.8: the reviewer (model_json) streams too now, so Stop ends it - told apart by its prompt.
                reviews = [r for r in run.requests if 'You verify whether' in r['messages'][0]['content']]
                self.assertEqual(bool(reviews), level in ('high', 'extra', 'max', 'ultra'))
                seen[level] = (started['limits']['steps_local'], calls[0]['max_tokens'])
        self.assertGreater(seen['max'][0], seen['low'][0])
        self.assertGreater(seen['max'][1], seen['low'][1])

    def test_zero_reasoning_room_override_is_saved_and_reaches_the_local_request(self):
        self.select('max')
        response = self.client.post('/api/settings', json={
            'effort_overrides': {'scripted-model': {'max': {'reasoning_tokens': 0}}}})
        self.assertEqual(response.status_code, 200, response.text)
        saved = config.load_settings()
        self.assertEqual(saved['effort_overrides']['scripted-model']['max']['reasoning_tokens'], 0)
        selected = self.client.get('/api/effort?mode=local&model=scripted-model').json()
        self.assertEqual(next(x['limits'] for x in selected['levels'] if x['id'] == 'max')['reasoning_tokens'], 0)
        run = replay.run_fixture(self.fixture(saved))
        self.assertEqual(run.status, 'success', run.message)
        calls = [request for request in run.requests if _agent_step(request)]
        self.assertTrue(calls)
        for call in calls:
            self.assertTrue(call['chat_template_kwargs']['enable_thinking'])
            self.assertEqual(call['max_tokens'], config.LOCAL_MAX_TOKENS)

    def test_real_settings_post_changes_running_limits_at_the_next_boundary(self):
        self.select('low')
        fx = self.fixture(config.load_settings())
        fx['setup']['settings_at_response'] = {'1': lambda: self.select('max')}
        run = replay.run_fixture(fx)
        self.assertEqual(run.status, 'success', run.message)
        changed = next(e for e in run.events if e.get('phase') == 'effort_changed')
        self.assertEqual(changed['effort'], 'max')
        self.assertEqual(changed['limits'], effort.PROFILES['max'].limits())
        calls = [r for r in run.requests if r.get('streamed')]
        self.assertFalse(calls[0]['chat_template_kwargs']['enable_thinking'])
        self.assertFalse(calls[1]['chat_template_kwargs']['enable_thinking'])
        self.assertTrue(calls[2]['chat_template_kwargs']['enable_thinking'])
        self.assertEqual(calls[2]['max_tokens'], config.LOCAL_MAX_TOKENS + effort.PROFILES['max'].reasoning_tokens)

    def test_session_choice_changes_the_next_run_without_changing_saved_settings(self):
        self.select('low')
        before = self.path.read_bytes()
        self.select('max', session=True)
        self.assertEqual(self.path.read_bytes(), before)
        run = replay.run_fixture(self.fixture(config.load_settings()))
        started = self.started(run)
        self.assertEqual((started['effort'], started['source']), ('max', 'session'))
        self.assertEqual(started['limits'], effort.PROFILES['max'].limits())

    def test_local_max_does_not_pretend_a_second_model_was_called(self):
        # The local runtime intentionally has one model at a time; don't imply
        # that a second-model review is guaranteed by the Max label.
        with mock.patch.object(models.llm_compat, 'completion') as complete:
            self.assertIsNone(models.model_json('verify', 'result', secondary=True))
            complete.assert_not_called()

    def test_cross_check_save_does_not_change_effort_and_failed_saves_preserve_disk(self):
        self.select('low')
        before = self.path.read_bytes()
        with mock.patch('atomic_io.replace', side_effect=OSError('Scripted disk failure')):
            response = self.client.post('/api/settings', json={'cross_check': True})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(shared.app_state['settings']['cross_check'])
        response = self.client.post('/api/settings', json={'cross_check': True})
        self.assertEqual(response.status_code, 200, response.text)
        saved = config.load_settings()
        self.assertTrue(saved['cross_check'])
        self.assertEqual(saved['effort_by_model']['scripted-model'], 'low')
        selected = self.client.get('/api/effort?mode=local&model=scripted-model').json()
        self.assertTrue(selected['cross_check']['enabled'])
        self.assertFalse(selected['cross_check']['available'])
        self.assertEqual(selected['applied'], 'low')
        run = replay.run_fixture(self.fixture(saved))
        self.assertEqual(run.result['verification']['cross_check']['status'], 'unavailable')
        self.assertTrue(any(e.get('phase') == 'cross_check_unavailable' for e in run.events))

    def test_second_model_availability_excludes_the_actual_model_and_duplicate_keys(self):
        with mock.patch.object(models, 'candidate_pairs', return_value=[('cloud/a', 'key1'), ('cloud/a', 'key2')]):
            self.assertFalse(models.cross_check_status(config.MODE_CLOUD, 'cloud/a')['available'])
        with mock.patch.object(models, 'candidate_pairs', return_value=[('cloud/a', 'key1'), ('cloud/b', 'key2')]), \
             mock.patch.object(models, '_current_model', return_value='cloud/b'), \
             mock.patch.object(shared, '_cooldown_until', {}):
            self.assertEqual(models.cross_check_status(config.MODE_CLOUD)['model'], 'cloud/a')
            shared.app_state['mode'] = config.MODE_CLOUD
            response = replay.as_response({'content': '{"verdict":"complete","findings":[]}'})
            with mock.patch.object(models.llm_compat, 'completion', return_value=response) as complete:
                self.assertEqual(models.model_json('verify', 'result', secondary=True)['verdict'], 'complete')
                self.assertEqual(complete.call_args.kwargs['model'], 'cloud/a')

    def test_cross_check_missing_defaults_to_off_and_null_updates_are_rejected(self):
        self.assertFalse(config.clean_agent_settings({'effort': 'max'})['cross_check'])
        before = self.path.read_bytes()
        response = self.client.post('/api/settings', json={'cross_check': None})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
