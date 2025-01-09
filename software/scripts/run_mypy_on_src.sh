#!/bin/bash

source venv/bin/activate

mypy --follow-imports=skip --strict --show-traceback $1