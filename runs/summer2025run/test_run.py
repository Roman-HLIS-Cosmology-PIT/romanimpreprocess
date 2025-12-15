# this is just to call solid-waffle from the command line
import sys

from solid_waffle.correlation_run import run_ir_all

run_ir_all(sys.argv[1])
