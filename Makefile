.PHONY: verify-phase1
verify-phase1:
	python3 -m pytest -q -m 'phase1 or global_gate'

