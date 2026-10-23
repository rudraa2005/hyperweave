.PHONY: verify-phase1 verify-phase2
verify-phase1:
	python3 -m pytest -q -m 'phase1 or global_gate'

verify-phase2:
	python3 -m pytest -q -m 'phase1 or phase2 or global_gate'

