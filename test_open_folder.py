"""POST /api/local-models/open-folder: opens the models folder on Windows, answers a clean error elsewhere."""
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import api.server as server
import core.config as config
import main


class OpenModelsFolderTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app, base_url="http://127.0.0.1:8000")

    def test_off_windows_it_is_a_clean_error_not_a_500(self):
        with patch.object(config, "IS_WINDOWS", False), patch.object(server.paths, "model_dirs", return_value=["/models"]):
            res = self.client.post("/api/local-models/open-folder", json={})
        self.assertEqual(res.status_code, 501)
        body = res.json()
        self.assertFalse(body["ok"])
        self.assertEqual(body["folder"], "/models")
        self.assertIn("Windows", body["message"])

    def test_on_windows_it_opens_the_models_folder(self):
        with patch.object(config, "IS_WINDOWS", True), patch.object(server.paths, "model_dirs", return_value=["/models"]), \
             patch.object(server.os, "makedirs"), patch.object(server.os, "startfile", create=True) as startfile:
            res = self.client.post("/api/local-models/open-folder", json={})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"ok": True, "folder": "/models"})
        startfile.assert_called_once_with("/models")


if __name__ == "__main__":
    unittest.main()
