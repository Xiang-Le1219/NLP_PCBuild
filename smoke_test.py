"""Offline deployment smoke test (no OPT downloads needed)."""
import advisor as A

engine = A.load_engine()
for text in ['Gaming PC below RM4500 with Ryzen and 16GB RAM',
             'Nak PC untuk kerja pejabat, bajet RM3000',
             'Need a coding PC, budget USD1000 with 1TB SSD',
             'Gaming lah, budget RM5000, include monitor and mouse']:
    req, source, raw = A.baseline_extract(text)
    result, violations = A.recommend(engine, req)
    assert result['ok'], (text, result.get('message'))
    assert not violations, (text, violations)
    assert result['total_myr'] <= result['budget_myr'] + .01
    print(f"PASS: {text} -> RM{result['total_myr']:.2f}")
req, _, _ = A.baseline_extract('I need a gaming PC')
result, _ = A.recommend(engine, req)
assert not result['ok']
print('PASS: missing budget rejected')
req, _, _ = A.baseline_extract('Gaming PC for RM1')
result, _ = A.recommend(engine, req)
assert not result['ok']
print('PASS: infeasible budget rejected')
print('All offline smoke tests passed. OPT and Streamlit hosting are not covered.')
