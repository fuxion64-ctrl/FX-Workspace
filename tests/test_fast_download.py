"""fast_download.py (1.4.6): pieces over several connections, resume from the piece record or an old one-stream .part,
and one plain stream when the server ignores Range. A fake server (httpx.MockTransport) - no network."""
import os
import shutil
import tempfile
import threading
import unittest
from unittest import mock

import httpx

import fast_download as fd

DATA = bytes(range(256)) * (4 * 1024 + 37)       # ~1 MB, not a multiple of a piece


def server(ranges=True, seen=None):
    def handle(req):
        h = req.headers.get("Range")
        if seen is not None:
            seen.append(h)
        if ranges and h:
            a, b = h.split("=")[1].split("-")
            a, b = int(a), int(b)
            return httpx.Response(206, content=DATA[a:b + 1], headers={"Content-Range": f"bytes {a}-{b}/{len(DATA)}"})
        return httpx.Response(200, content=DATA)
    return lambda: httpx.Client(transport=httpx.MockTransport(handle))


class FastDownloadTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.tmp = os.path.join(self.dir, "m.gguf.part")
        self.addCleanup(shutil.rmtree, self.dir, True)
        p = mock.patch.multiple(fd, PIECE=64 * 1024, MIN_SPLIT=1024, SAVE_EVERY=16 * 1024)
        p.start()
        self.addCleanup(p.stop)

    def read(self):
        with open(self.tmp, "rb") as f:
            return f.read()

    def test_pieces_over_several_connections_make_the_whole_file(self):
        seen, got = [], []
        with mock.patch.object(fd, "_client", server(seen=seen)):
            fd.fetch("https://x/m", self.tmp, len(DATA), progress=got.append, conns=4)
        self.assertEqual(self.read(), DATA)
        self.assertEqual(sum(got), len(DATA))
        self.assertGreater(len([h for h in seen if h and h != "bytes=0-0"]), 4)    # many ranged requests
        self.assertFalse(os.path.exists(self.tmp + ".json"))                       # the record goes when done

    def test_a_cancelled_download_resumes_where_each_piece_stopped(self):
        cancel, n = threading.Event(), [0]

        def prog(k):
            n[0] += k
            if n[0] > len(DATA) // 3:
                cancel.set()
        with mock.patch.object(fd, "_client", server()):
            with self.assertRaises(fd.Cancelled):
                fd.fetch("https://x/m", self.tmp, len(DATA), cancel=cancel, progress=prog, conns=2)
            self.assertTrue(os.path.exists(self.tmp + ".json"))
            seen = []
            with mock.patch.object(fd, "_client", server(seen=seen)):
                fd.fetch("https://x/m", self.tmp, len(DATA), conns=2)
        self.assertEqual(self.read(), DATA)
        asked = sum(int(h.split("-")[1]) - int(h.split("=")[1].split("-")[0]) + 1 for h in seen if h and h != "bytes=0-0")
        self.assertLess(asked, len(DATA))                                          # only what was missing

    def test_an_old_one_stream_part_counts_as_its_first_bytes(self):
        with open(self.tmp, "wb") as f:
            f.write(DATA[:200_000])
        seen = []
        with mock.patch.object(fd, "_client", server(seen=seen)):
            fd.fetch("https://x/m", self.tmp, len(DATA), conns=3)
        self.assertEqual(self.read(), DATA)
        self.assertFalse(any(h == "bytes=0-65535" for h in seen))                 # the first piece was already there

    def test_a_server_without_ranges_gets_one_plain_stream(self):
        with mock.patch.object(fd, "_client", server(ranges=False)):
            fd.fetch("https://x/m", self.tmp, len(DATA), conns=4)
        self.assertEqual(self.read(), DATA)

    def test_tor_uses_fewer_connections(self):
        with mock.patch.object(fd.netlayer, "route_for", return_value=mock.Mock(kind="tor")):
            self.assertEqual(fd.connections(), fd.TOR_CONNECTIONS)


if __name__ == "__main__":
    unittest.main()
