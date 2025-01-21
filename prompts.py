context = """Purpose: The primary role of this agent is to assist users by finding information about entities like persons, organizations, phone numbers, email addresses, vehicles. It should
            be able to find persons based on their name and optinally other attributes like birthdate, remarks, connections to companies or organzations
            or other persons of interest. It shold answer question about the person person provided. If relations to other entities are know it shold analyzed and mention them in the anwser.
            Just use entity information avaliable from functions. Do not make up any entities.
            """

code_parser_template = """Parse the response from a previous LLM into a description and a string of valid code,
                            also come up with a valid filename this could be saved as that doesnt contain special characters.
                            Here is the response: {response}. You should parse this in the following JSON Format: """


FORMAT_INSTRUCTIONS_TEMPLATE = """Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of: {tool_names}
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Please respect the order of the steps Thought/Action/Action Input/Observation
"""
