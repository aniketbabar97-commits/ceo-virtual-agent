#!/usr/bin/env python3
"""
Real Groq testing script.
Runs 10 test cycles with actual Groq API to validate fixes:
- No 'Manager agent should not have tools'
- No cache_breakpoint errors
- Successful crew kickoff at least in planning phase
"""
import os
import sys
import time
sys.path.insert(0, '/home/user/ceo-repo')

from ceo_core import run_one_ceo_cycle, get_config

# Key should be provided via environment variable for security
# os.environ["GROQ_API_KEY"] = "YOUR_KEY_HERE"  # Do not hardcode in committed code

print("=" * 70)
print("REAL GROQ END-TO-END TESTS (10 cycles)")
print("Testing fixes for Manager tools + Groq cache_breakpoint")
print("=" * 70)

config = {
    "provider": "groq",
    "api_key": os.environ["GROQ_API_KEY"],
    "model": "llama-3.3-70b-versatile",
    "autonomous_interval_minutes": 60,
    "max_cycles_per_run": 1
}

goal = "Build a profitable faceless content business (newsletter, blog, or YouTube-style content) in a high-demand niche like personal finance, productivity, health/fitness, or AI/tools for beginners. Monetize primarily via affiliate links, digital products."

successes = 0
manager_tools_errors = 0
cache_errors = 0
other_errors = 0

for i in range(1, 11):
    print(f"\n--- Real Test Cycle {i}/10 ---")
    try:
        report = run_one_ceo_cycle(config, goal=goal, max_retries=1)
        
        if report.get("success"):
            print("  ✓ SUCCESS: Cycle completed without critical errors")
            successes += 1
        else:
            error = report.get("error", "")
            print(f"  Partial: {error[:200]}")
            
            if "Manager agent should not have tools" in error:
                manager_tools_errors += 1
            elif "cache_breakpoint" in error or "unsupported" in error.lower():
                cache_errors += 1
            else:
                other_errors += 1
                
    except Exception as e:
        error_str = str(e)
        print(f"  ✗ Exception: {error_str[:300]}")
        
        if "Manager agent should not have tools" in error_str:
            manager_tools_errors += 1
        elif "cache_breakpoint" in error_str or "unsupported" in error_str.lower():
            cache_errors += 1
        else:
            other_errors += 1
    
    time.sleep(2)  # Rate limit respect

print("\n" + "=" * 70)
print("FINAL TEST RESULTS (10 real Groq cycles)")
print(f"Full successes: {successes}/10")
print(f"Manager tools errors: {manager_tools_errors}")
print(f"Cache_breakpoint / Groq unsupported errors: {cache_errors}")
print(f"Other errors: {other_errors}")
print("=" * 70)

if manager_tools_errors == 0 and cache_errors == 0:
    print("✅ All critical fixes validated with real Groq API!")
else:
    print("⚠️ Some issues remain - need more fixes before push.")