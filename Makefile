.PHONY: verify-phase1 verify-phase2 verify-phase3 verify-phase4 verify-phase5
verify-phase1:
	python3 -m pytest -q -m 'phase1 or global_gate'

verify-phase2:
	python3 -m pytest -q -m 'phase1 or phase2 or global_gate'

verify-phase3:
	python3 -m pytest -q -m 'phase1 or phase2 or phase3 or global_gate'

verify-phase4:
	python3 -m pytest -q -m 'phase1 or phase2 or phase3 or phase4 or global_gate'

verify-phase5:
	python3 -m pytest -q -m 'phase1 or phase2 or phase3 or phase4 or phase5 or global_gate'

