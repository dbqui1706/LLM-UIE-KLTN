USER_PROMPTS = {
    "NER": """Extract entities from the text **strictly using ONLY the provided Entity List** below and **MUST** strictly adhere to the output format.
Format output as '<entity tag>: <entity name>' and separated multiple entities by '|'. Return 'None' if no entities are identified.
Entity List: {ner_labels}
Text: {text}""",

    "RE": """Extract relationships between entities in text **strictly using ONLY the provided Relationship List** below and **MUST** strictly adhere to the output format.
Format each relationship as '<relation_type>: <head_entity>, <tail_entity>' and separated multiple relationship by '|'. Return 'None' if no relationships are identified.
Relationship List: {re_labels}
Text: {text}""",

    "EE": """Extract events and their components from text **strictly using ONLY the provided Event List** below and **MUST** strictly adhere to the output format.
Format output as '<event_type>: <trigger_word> | <role1>: <argument1> | <role2>: <argument2>' and separate multiple events with '|'. Return 'None' if no events are identified.
Event List: {ee_labels}
Text: {text}"""
}