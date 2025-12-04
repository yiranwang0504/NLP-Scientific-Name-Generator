import google.generativeai as genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
genai.configure(api_key='AIzaSyB4S9q5VRFn5wV-8WnTBttNu0kpGeG5vHY')
model = genai.GenerativeModel("gemini-2.5-flash")

response = model.generate_content("""Could you just give me Latin names of the description below? example_prompts = [
    "Description: a large brown bear with a scar on its paw\nFamily: Ursidae\nName:",
    "Description: a tiny gray mouse living in a barn\nFamily: Muridae\nName:",
    "Description: a colorful parrot that can imitate human speech\nFamily: Psittacidae\nName:",
    "Description: a dark green frog that lives near waterfalls\nFamily: Ranidae\nName:",
    "Description: a fast-running desert fox\nFamily: Canidae\nName:",
    "Description: a golden-scaled fish often seen in garden ponds\nFamily: Cyprinidae\nName:",
    "Description: a fluffy black rabbit with long ears\nFamily: Leporidae\nName:",
    "Description: a snow owl known for silent flight\nFamily: Strigidae\nName:",
    "Description: a gentle giant elephant with long tusks\nFamily: Elephantidae\nName:",
    "Description: a red-striped tiger wandering in bamboo forests\nFamily: Felidae\nName:",
    "Description: a shy hedgehog that curls into a ball\nFamily: Erinaceidae\nName:",
    "Description: a sleek black panther that hunts at night\nFamily: Felidae\nName:",
    "Description: a curious dolphin that plays with seaweed\nFamily: Delphinidae\nName:",
    "Description: a slow-moving turtle with a patterned shell\nFamily: Testudinidae\nName:",
    "Description: a bright green lizard sunbathing on warm rocks\nFamily: Lacertidae\nName:"
]""")

print(response.text)