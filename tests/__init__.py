import asyncio
import os
import unittest
from typing import Dict, List

from gradescope_utils.autograder_utils.files import SUBMISSION_BASE

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
