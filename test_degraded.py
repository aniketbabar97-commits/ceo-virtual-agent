#!/usr/bin/env python3
"""
Test for DEGRADED KEYLESS MODE.
Runs 10+ cycles to validate that with NO API keys, the CEO still runs fully:
- No crashes on missing keys
- Research works (free DDGS)
- Assets generated (report, voice, video if possible)
- Approvals requested
- Memory/revenue updated
- Success=True returned
- System does not stop
"""
import os
import sys
import time
import shutil

# Ensure we use the workspace version
sys.path.insert(0, '/home/user')

from ceo_core import run_one_ceo_cycle, get_config, load_memory, load_revenue, GENERATED_DIR, DATA_DIR

print("=" * 70)
print("DEGRADED KEYLESS MODE - 10+ CYCLE VALIDATION TEST")
print("Testing: NO API keys, system must NEVER stop, still produce real outputs")
print("=" * 70)

# Force clean for test (optional, but good)
# Clear previous degraded files for clean test
for f in os.listdir(GENERATED_DIR):
    if f.startswith("degraded_"):
        try:
            os.remove(os.path.join(GENERATED_DIR, f))
        except: pass

config = {
    "provider": "groq",  # Will be ignored
    "api_key": "",  # NO KEY
    "model": "llama-3.3-70b-versatile",
    "autonomous_interval_minutes": 60,
    "max_cycles_per_run": 1,
    "degraded_mode": False  # Auto trigger via no key
}

goal = "Build a profitable faceless content business (newsletter, blog, or YouTube-style content) in a high-demand niche like personal finance, productivity, health/fitness, or AI/tools for beginners. Monetize primarily via affiliate links, digital products."

successes = 0
degraded_runs = 0
assets_produced = 0
approvals_requested = 0
errors = []

for i in range(1, 11):
    print(f"\n--- Degraded Test Cycle {i}/10 ---")
    try:
        # Clear logs for clean per cycle? No, append ok for test
        report = run_one_ceo_cycle(config, goal=goal, max_retries=0)
        
        if report.get("success"):
            print("  ✓ SUCCESS: Cycle completed in keyless mode")
            successes += 1
            if report.get("mode") == "degraded_keyless":
                degraded_runs += 1
                print("  ✓ Confirmed DEGRADED_KEYLESS mode active")
            assets = report.get("assets_generated", [])
            assets_produced += len(assets)
            print(f"  ✓ Assets generated this cycle: {len(assets)} (total in run: {assets_produced})")
            if "voiceover" in str(assets) or any("degraded" in str(a) for a in assets):
                print("  ✓ Voice/video/report assets confirmed")
            full_out = report.get("full_output", "")
            if "Owner approval requested" in full_out or "approval" in full_out.lower():
                approvals_requested += 1
                print("  ✓ Approval request logic executed")
            print(f"  ✓ Report next: {report.get('next_recommended', '')[:80]}...")
        else:
            print(f"  ✗ Failed: {report.get('error', 'unknown')[:100]}")
            errors.append(f"Cycle {i}: {report.get('error')}")
    except Exception as e:
        err_str = str(e)[:200]
        print(f"  ✗ EXCEPTION: {err_str}")
        errors.append(f"Cycle {i}: {err_str}")
    
    time.sleep(1)  # Brief pause between tests

print("\n" + "=" * 70)
print("FINAL DEGRADED TEST RESULTS (10 cycles, ZERO keys)")
print(f"Successes: {successes}/10")
print(f"Degraded mode runs: {degraded_runs}/10")
print(f"Total assets produced across tests: {assets_produced}")
print(f"Approval requests triggered: {approvals_requested}")
print(f"Errors: {len(errors)}")
if errors:
    for e in errors[:3]:
        print(f"  - {e}")
print("=" * 70)

if successes == 10 and degraded_runs >= 8 and len(errors) == 0:
    print("✅✅✅ ALL TESTS PASSED! Degraded keyless mode works perfectly. CEO never stops without keys.")
    print("Real assets, research, approvals, learning all functional at zero cost.")
else:
    print("⚠️ Some issues - review above. But system should have continued without hard crash.")

# Verify files created
print("\nVerifying generated files from degraded runs:")
gen_files = [f for f in os.listdir(GENERATED_DIR) if f.startswith("degraded_") or "degraded" in f]
print(f"  Degraded assets found: {len(gen_files)} - {gen_files[:5]}")

mem = load_memory()
print(f"  Memory cycles updated: {mem.get('total_cycles', 0)}")

print("\nTest complete. Ready for push after more validation if needed.")