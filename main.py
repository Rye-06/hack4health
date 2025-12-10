# Questions that help narrow down the user's risk
import sys 
from familyRiskPrediction import family_increased_risk

def _ask_yes_no(prompt):
    while True:
        response = input(prompt + " (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Please answer 'yes' or 'no'.")

def _ask_int(prompt, default=0):
    while True:
        s = input(f"{prompt} (enter integer, default {default}): ").strip()
        if s == "":
            return int(default)
        try: 
            v = int(s) 
            if v < 0:
                print("Please enter a positive integer.")
                continue
            return v
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

def run_risk_assessement():
    # fixed baseline risk
    BASELINE_RISK = 0.10 # 10% baseline

    print("Family Alzheimer's risk calculator\n")
    print(f"Using baseline risk = {BASELINE_RISK*100:.1f}%\n")

    # First degree: mother, father, brothers/sisters
    mother = _ask_yes_no("Does your mother have Alzheimer's?")
    father = _ask_yes_no("Does your father have Alzheimer's?")
    siblings_count = _ask_int("How many of your siblings have Alzheimer's?", default=0)

    first_degree_count = int(mother) + int(father) + siblings_count

    #Second degree: cousines, aunts/uncles, grandparents
    grandparents_count = _ask_int("How many of your grandparents have Alzheimer's?", default=0)
    aunts_uncles_count = _ask_int("How many of your aunts/uncles have Alzheimer's?", default=0)
    cousins_count = _ask_int("How many of your cousins have Alzheimer's?", default=0)

    second_degree_count = int(grandparents_count) + int(aunts_uncles_count) + int(cousins_count)

    new_risk, breakdown = family_increased_risk(
        BASELINE_RISK,
        first_degree_count,
        second_degree_count
    )

    print("\nRisk assessment results:")
    print(f" First-degree multiplier: {breakdown.get('risk_first', 1.0):.2f}")
    print(f" Second-degree multiplier: {breakdown.get('risk_second', 1.0):.2f}")
    print(f" Combined risk multiplier: {breakdown.get('overall_risk', 1.0):.2f}")
    print(f"\n Estimated Azhiemer's risk: {new_risk*100:.2f}%")

if __name__ == "__main__":
    try: 
        run_risk_assessement()
    except KeyboardInterrupt:
        print("\nAssessment cancelled.")
        sys.exit(0)

