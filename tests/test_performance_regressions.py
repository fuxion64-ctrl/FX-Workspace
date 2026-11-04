"""Local inference speed, timing and lifecycle regressions (no real model)."""
import threading
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import local_runtime
import model_metrics
import system_metrics


class TimingTests(unittest.TestCase):
    def test_server_decode_timing_wins_over_request_duration(self):
        response = NS(timings={'predicted_n': 100, 'predicted_ms': 5000},
                      usage=NS(completion_tokens=100), stream_timing={'generation_s': 30})
        facts = model_metrics.generation_facts(response)
        speed = model_metrics.aggregate_speed([{**facts, 'duration_ms': 300000, 'first_token_s': 20}])
        self.assertEqual(speed['tps'], 20)
        self.assertEqual(speed['speed_source'], 'server')

    def test_untimed_helper_does_not_dilute_streamed_generation(self):
        speed = model_metrics.aggregate_speed([
            {'tokens_out': 100, 'duration_ms': 15000, 'first_token_s': 10},
            {'tokens_out': 20, 'duration_ms': 90000}])
        self.assertEqual(speed['tps'], 20)
        self.assertEqual(speed['speed_tokens'], 100)

    def test_unknown_generation_time_does_not_claim_tokens_per_second(self):
        self.assertEqual(model_metrics.aggregate_speed([{'tokens_out': 100, 'duration_ms': 300000}]), {})
        self.assertEqual(model_metrics.generation_facts(NS(timings={'predicted_n': 10, 'predicted_ms': float('nan')})), {})

    def test_stream_interval_excludes_first_token_and_uses_usage_not_chunks(self):
        facts = model_metrics.generation_facts(NS(timings=None, usage=NS(completion_tokens=101),
                                                 stream_timing={'generation_s': 5}))
        self.assertEqual(facts['generation_tokens'], 100)
        self.assertEqual(model_metrics.aggregate_speed([facts])['tps'], 20)

    def test_server_rates_are_weighted_by_time(self):
        speed = model_metrics.aggregate_speed([
            {'generation_tokens': 100, 'generation_s': 5, 'speed_source': 'server'},
            {'generation_tokens': 30, 'generation_s': 1, 'speed_source': 'server'}])
        self.assertEqual(speed['tps'], 21.7)


class RuntimeTests(unittest.TestCase):
    def test_lifecycle_reservation_refuses_a_parallel_restart(self):
        held, release = threading.Event(), threading.Event()
        def inference():
            with local_runtime.reserve():
                held.set()
                release.wait(3)
        worker = threading.Thread(target=inference)
        worker.start()
        try:
            self.assertTrue(held.wait(2))
            self.assertTrue(local_runtime.busy())
            with self.assertRaises(RuntimeError), local_runtime.reserve(wait=False):
                self.fail('Acquired the active inference reservation')
        finally:
            release.set()
            worker.join(3)
        self.assertFalse(local_runtime.busy())

    def test_nested_load_is_reentrant_and_exception_releases(self):
        with self.assertRaises(ValueError), local_runtime.reserve():
            with local_runtime.reserve(wait=False):
                raise ValueError('load failed')
        self.assertFalse(local_runtime.busy())

    def test_resident_model_does_not_run_nvidia_smi_for_every_call(self):
        with patch.dict(system_metrics._MODEL, baseline_used_mb=100, name='model', estimated_mb=4000), \
                patch.object(system_metrics, '_gpu_sample') as sample:
            system_metrics.begin_model_load()
            system_metrics.finish_model_load('model')
        sample.assert_not_called()


class ContextAndApiTests(unittest.TestCase):
    def test_local_fit_and_meter_use_the_model_window_not_the_effort_budget(self):
        # 1.4.1 keeps 1.4.0's behaviour by the user's choice: history fills the model's window.
        import agent.models as llm
        import agent.context_window as cw
        import core.state as shared
        profile = NS(context_tokens=6000, name='quick')
        with patch.dict(shared.app_state, conversation=[], run_budget=None), \
                patch.object(cw, '_profile', return_value=profile), \
                patch.object(llm, 'resolve_local_model', return_value='test'), \
                patch.object(llm.llama_local, 'model_context_limit', return_value=80000), \
                patch.object(llm.schemas, 'tools_for_current_turn', return_value=[]), \
                patch.object(llm, 'system_prompt', return_value='system'), \
                patch.object(llm, '_turn_tokens', return_value=0), \
                patch.object(llm, '_local_reply_tokens', return_value=1000), \
                patch.object(cw, '_tools_tokens', return_value=2000), \
                patch.object(llm, 'trim_conversation') as trim:
            # _tools_tokens pinned (4 Nov): the tool list's size depended on which tools earlier tests left on, which
            # made this fail in some full runs (69,936) - the point here is the 80K window, not the tools.
            llm._fit_for_local()
            budget = trim.call_args.kwargs['budget_tokens']
            self.assertGreater(budget, 70000)
            self.assertGreater(cw.status('test', True)['history_limit'], 70000)

    def test_hidden_local_helper_leaves_the_replys_live_stream_alone(self):
        # A reply may set its stream id while a hidden helper holds the model; the helper must not reset it.
        import agent.models as llm
        import core.state as shared
        seen = {}
        def complete(kwargs, record=True, publish=True):
            seen['publish'] = publish
            shared.app_state['_stream_call'] = 'reply-call'      # the reply registers meanwhile
            return NS(choices=[NS(message=NS(content='ok'))], usage=None)
        with patch.dict(shared.app_state, _stream_call=None, running_model_event=None), \
                patch.object(llm, '_ensure_build'), patch.object(llm, '_watch_load', return_value=None), \
                patch.object(llm, 'engage_local_model'), patch.object(llm, '_local_complete', side_effect=complete):
            out = llm.model_text('system', 'user', model='local:test')
            self.assertEqual(out['text'], 'ok')
            self.assertFalse(seen['publish'])
            self.assertEqual(shared.app_state['_stream_call'], 'reply-call')

    def test_partial_resource_update_keeps_unmentioned_offload_options(self):
        import api.server as server
        with patch.object(server.llama_local, 'update_model_config') as update, \
                patch.object(server.llama_local, 'list_model_configs', return_value=[]), \
                patch.object(server.model_scan, 'restart_router', return_value=False), \
                patch.object(server.event_log, 'log_action'):
            server.set_local_model_config(server.ModelConfigUpdate(name='m', ctx_size=8192))
            update.assert_called_once_with('m', ctx_size=8192)
            update.reset_mock()
            server.set_local_model_config(server.ModelConfigUpdate(name='m', ngl=None))
            update.assert_called_once_with('m', ngl=None)


if __name__ == '__main__':
    unittest.main()
