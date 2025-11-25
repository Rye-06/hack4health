def family_increased_risk(base_risk, first_degree_count, second_degree_count, 
                          first_degree_risk_factor=2.0, second_degree_risk_factor=1.5):
    """
    Calculate the increased risk of alzheimer's disease based on family history."""
    
    #validates inputs
    if base_risk < 0:
        raise ValueError("Base risk must be greater than or equal to 0.")
    

    try:
        first_degree_count = int(first_degree_count)
        second_degree_count = int(second_degree_count)
    except Exception:
        raise ValueError("Family member counts must be integers")
    

    if first_degree_count < 0 or second_degree_count < 0:
        raise ValueError("Family member count must be greater than or equal to 0")
    
    