PYTHON ?= python

.PHONY: test demos check
test:
	$(PYTHON) -m unittest discover -s tests -v

demos:
	$(PYTHON) -m src.app.cli batch input --write-reports

check: test
	$(PYTHON) -m src.app.cli batch input
