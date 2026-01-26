import signal
import sys
import time
import os
import threading
import traceback
import psutil

def dump_thread_info(signum, frame):
    """Signal handler to dump all thread stack traces"""
    with open(f"/tmp/process_{os.getpid()}_thread_dump.dump", 'w') as fp:

      print("\n" + "="*80, file=fp)
      print(f"THREAD DUMP - Signal {signum} received", file=fp)
      print("="*80, file=sys.stderr)

      for thread_id, frame in sys._current_frames().items():
          # Find the thread object
          thread = None
          for t in threading.enumerate():
              if t.ident == thread_id:
                  thread = t
                  break

          thread_name = thread.name if thread else f"Unknown-{thread_id}"
          is_daemon = thread.daemon if thread else "?"
          is_alive = thread.is_alive() if thread else "?"

          print(f"\n--- Thread: {thread_name} (ID: {thread_id}) ---", file=fp)
          print(f"    Daemon: {is_daemon}, Alive: {is_alive}", file=fp)
          print("    Stack trace:", file=fp)

          # Print the stack trace
          for filename, lineno, name, line in traceback.extract_stack(frame):
              print(f"      File: {filename}:{lineno}", file=fp)
              print(f"      Function: {name}", file=fp)
              if line:
                  print(f"      Code: {line}", file=fp)
          print(file=fp)

      print("="*80 + "\n", file=fp)

      for child in psutil.Process().children():
        print("=" * 80 + "\n")
        print(f" ---- Sending signal to child {child.pid} ---- ", file=fp)
        print(f"      Child is {child.status()} \n", file=fp)
        child.send_signal(signum)
        time.sleep(1.0)
        print("=" * 80 + "\n", file=fp)

print(f"Registered Debug signal handler! send to {os.getpid()}")
signal.signal(signal.SIGUSR1, dump_thread_info)