"""Smoke test 1 case on U2-RadiMed and 1 case on GPT-6 Luna."""
import json
from pathlib import Path
from run_remote_u2_400 import call_single as call_u2
from run_remote_luna_200 import call_single as call_luna

u2_cases = json.load(open("/home/eason/rsna-knee/cohort_400_u2_radimed.json"))
luna_cases = json.load(open("/home/eason/rsna-knee/cohort_200_gpt6_luna.json"))

print("Testing 1 case on U2-RadiMed...")
uid1, res1 = call_u2(u2_cases[0])
print(f"U2-RadiMed: {uid1} -> {list(res1.keys())}")
if "diagnoses" in res1:
    print("  Diagnoses found:", list(res1["diagnoses"].keys()))

print("\nTesting 1 case on GPT-6 Luna...")
uid2, res2 = call_luna(luna_cases[0])
print(f"GPT-6 Luna: {uid2} -> {list(res2.keys())}")
if "diagnoses" in res2:
    print("  Diagnoses found:", list(res2["diagnoses"].keys()))

print("\nSmoke test successfully passed!")
