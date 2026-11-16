PY ?= python
REPORTS ?= reports

.PHONY: test figures clean

test:
	$(PY) -m pytest

figures:
	$(PY) scripts/make_figures.py --exp dog --reports $(REPORTS)
	$(PY) scripts/make_comparison_figures.py --reports $(REPORTS)

clean:
	rm -rf $(REPORTS)/figures
