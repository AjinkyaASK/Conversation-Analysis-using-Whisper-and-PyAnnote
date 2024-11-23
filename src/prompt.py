QNA_PROMPT_MESSAGE = """
    <TRANSCRIPTION>
    
    - Above is the transcription of an interview.
    - Analyse the the communication and give me list of questions asked by interviewer and their answers given by the candidate.
    - Evaluate whether the answer is right, partially right or wrong. And you can put 'unknown' if you are not sure about correctness of the answer.
    - Put remark when a answer is not completely right.
    - Rate each 'right' and 'partially right' answer out of 10 (1 to 10).
    
    I want your response as a pure JSON, in following format:
    
    {
        "result": [
            {
                "question": "",
                "answer": "",
                "correctness": "",
                "remark": "",
                "rating": 1
            },
            {
                "question": "",
                "answer": "",
                "correctness": "",
                "remark": "",
                "rating": 1
            }
        ]
    }
    
    Make sure following things:
    1. Use less technical and simple to understand words in impact section.
    2. Apart from the JSON, no other text is present in your response.
    3. The response is in a valid JSON format.
    """