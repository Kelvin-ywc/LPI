from openai import OpenAI
api_key = "sk-MazpnWiEWQhrgtP8526a79F8D7254a5894296d2d81Ea6c7a"
api_base = "https://oneapi.xty.app/v1"
client = OpenAI(
    api_key=api_key,
    base_url=api_base
)
model = "text-embedding-3-large"


def get_embedding(text, model):
    text = text.replace("\n", " ")
    embeddings = client.embeddings
    creation = embeddings.create(input =[text], model=model)

    return creation.data[0].embedding


task_names =['appliance', 'sports', 'outdoor','electronic', 'accessory', 'indoor','kitchen', 'furniture', 'vehicle','food', 'animal', 'person']

task_senmantic_embedding =[get_embedding(task_name, model) for task_name in task_names]