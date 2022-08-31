# import time module, Observer, FileSystemEventHandler
import logging
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import policy_iteration2

class OnMyWatch:
	# Set the directory on watch
	watchDirectory = "./observations"

	def __init__(self):
		self.observer = Observer()

	def run(self):
		event_handler = Handler()
		self.observer.schedule(event_handler, self.watchDirectory, recursive = True)
		self.observer.start()
		try:
			while True:
				#print('a')
				logging.info('sleeping...')
				time.sleep(5)
		except:
			self.observer.stop()
			#print("Observer Stopped")
			logging.info('Observer Stopped.')

		self.observer.join()


class Handler(FileSystemEventHandler):

	@staticmethod
	def on_any_event(event):
		if event.is_directory:
			return None

		elif event.event_type == 'created':
			# Event is created, you can process it now
			#print("Watchdog received created event - % s." % event.src_path)
			what = 'directory' if event.is_directory else 'file'
			logging.info("Created %s: %s", what, event.src_path)
			file1 = open(event.src_path, "r")
			logging.info(file1.read())
		elif event.event_type == 'modified':
			# Event is modified, you can process it now
			#print("Watchdog received modified event - % s." % event.src_path)
			what = 'directory' if event.is_directory else 'file'
			logging.info("Modified %s: %s", what, event.src_path)
			if (not(event.is_directory)):
				file1 = open(event.src_path, "r")
				data = file1.read().split()
				observation = [ int(x) for x in data[0].split(',')]
				#observation = [0,0,1,-1]
				probability = data[1]
				observations.append((observation,probability))
				policy_iteration2.main_iterative(obs=observation)

if __name__ == '__main__':
	observations = []
	logging.basicConfig(level=logging.DEBUG)
	logging.info('Watchdog main function called.')
	watch = OnMyWatch()
	watch.run()
