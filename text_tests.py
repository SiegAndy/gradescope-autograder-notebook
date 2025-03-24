import unittest

unittest.TestLoader.sortTestMethodsUsing = None

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover("tests/PA3")
    unittest.TextTestRunner().run(suite)
