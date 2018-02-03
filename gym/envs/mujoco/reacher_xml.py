import xml.etree.cElementTree as ElementTree
import numpy as np
import os

MIN_LEN = 0.05
MAX_LEN = 0.2
MIN_FORCE = 1.00
MAX_FORCE = 1

class ReacherXML(object):
	def __init__(self):
		self.asset_dir = os.path.join(os.path.dirname(__file__), "assets")
		path = os.path.join(self.asset_dir, "reacher.xml")
		self.tree = ElementTree.parse(path)
		up = self.tree.find("worldbody/body")
		fore = up.find("body")
		self.fingertip = fore.find("body").attrib
		self.up = up.find("geom").attrib
		self.fore_attrib = fore.attrib
		self.fore_geom = fore.find("geom").attrib
		self.actuators = [motor.attrib for motor in
			self.tree.find('actuator').findall('motor')]

	def randomize(self, np_random):
		up_len, fore_len = np_random.uniform(MIN_LEN, MAX_LEN, size=(2,))
		fmt_str = "0 0 0 {} 0 0"
		self.up["fromto"] = fmt_str.format(up_len)
		self.fore_attrib["pos"] = "{} 0 0".format(up_len)
		self.fore_geom["fromto"] = fmt_str.format(fore_len)
		self.fingertip["pos"] = "{} 0 0".format(fore_len + 0.01)

		self.max_rad = up_len + fore_len
		self.min_rad = np.abs(up_len - fore_len)

		up_force, fore_force = np_random.uniform(MIN_FORCE, MAX_FORCE, size=(2,))
		self.actuators[0]["ctrlrange"] = "{} {}".format(-up_force, up_force)
		self.actuators[1]["ctrlrange"] = "{} {}".format(-fore_force, fore_force)

		self.sysid_vec = np.array([up_len, fore_len, up_force, fore_force])
		path = os.path.join(self.asset_dir, "reacher_randomized.xml")
		self.tree.write(path)

if __name__ == '__main__':
	r = ReacherXML()
	npr = np.random.RandomState()
	r.randomize(npr)
