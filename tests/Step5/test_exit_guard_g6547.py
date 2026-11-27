"""g6547 exit guard: a finished or stopped trainer whose exit is blocked (a worker thread that never ends) is ended
within the grace period, after writing its thread stacks, with the exit code it would have had."""
import subprocess
import sys
import textwrap
import time
import unittest

from brass_ai.learning.contracts import SOURCE_ROOT

CHILD = textwrap.dedent('''
    import sys, threading, time
    sys.path.insert(0, {src!r})
    from brass_ai.step5 import exit_guard
    exit_guard.GRACE_SECONDS = 1.
    def run():
        threading.Thread(target=threading.Event().wait, name='blocked-join').start()  # never ends: exit would wait
        if {fail!r}: raise RuntimeError('training failed')
        return {code!r}
    exit_guard.guarded(run)
''')


def child(code=0, fail=False):
    source = CHILD.format(src=str(SOURCE_ROOT / 'src'), code=code, fail=fail)
    began = time.monotonic()
    done = subprocess.run([sys.executable, '-c', source], capture_output=True, text=True, timeout=60)
    return done, time.monotonic() - began


class ExitGuard(unittest.TestCase):
    def test_a_finished_command_with_a_blocked_thread_still_exits(self):
        done, seconds = child(code=0)
        self.assertEqual(done.returncode, 0)
        self.assertLess(seconds, 30)
        self.assertIn('thread stacks follow', done.stderr)
        self.assertIn('blocked-join', done.stderr)  # the stacks name what blocked the exit

    def test_a_failed_command_exits_with_a_failure_code(self):
        done, seconds = child(fail=True)
        self.assertEqual(done.returncode, 1)
        self.assertIn('training failed', done.stderr)
        self.assertLess(seconds, 30)


if __name__ == '__main__':
    unittest.main()
