from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from contants import embeddings_path, EMBEDDING_MODEL, GPT_MODEL

from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from scipy import spatial
import ast

load_dotenv()
client = OpenAI()

df = pd.read_csv(embeddings_path)

# convert embeddings from CSV str type back to list type
df['embeddings'] = df['embeddings'].astype(str)
df['embeddings'] = df['embeddings'].apply(ast.literal_eval)

relatedness_fn = lambda x, y: 1 - spatial.distance.cosine(x, y)

# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=relatedness_fn,
    top_n: int = 2
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (row["ticket"], relatedness_fn(query_embedding, row["embeddings"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]


def get_similar_score(base_query, base_contexts, relatedness_fn):
    query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=base_query,
    )

    query_embedding = query_embedding_response.data[0].embedding

    similar_contexts = []

    for context in base_contexts:
        context_embedding_response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=context,
        )

        context_embedding = context_embedding_response.data[0].embedding

        relatedness = relatedness_fn(query_embedding, context_embedding)

        if relatedness>0.80:
            print("Score: ", relatedness)
            similar_contexts.append(context)

    return similar_contexts


def ask(
    query: str,
    model: str = GPT_MODEL,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT."""
    message = query
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "Hi."},
        {"role": "user", "content": message},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        # temperature=0,
        max_tokens=512
    )
    response_message = response.choices[0].message.content
    return response_message


# initialize FastAPI entry point
app = FastAPI()

# Enable CORS for all origins (replace '*' with specific origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

@app.post('/api/similar-tickets')
async def similar_tickets(query: Query):
    try:
        result, relatedeness = strings_ranked_by_relatedness(query=query.query, df=df, top_n=3)

        tickets = []

        for string, relatedeness in zip(result, relatedeness):
            idx = df.ticket.tolist().index(string)

            case_ids = df['case_id'].tolist()
            case_nums = df['case_number'].tolist()
            
            case_id = case_ids[idx]
            case_number = case_nums[idx]
            # break_pt = tick.split("Description:")
            # subject = break_pt[0][9:].strip()
            # description = break_pt[1].strip()

            ticket = {
                "case_id": case_id,
                "case_number": case_number,
                # "subject": subject,
                # "description": description,
                "score": relatedeness
            }
            tickets.append(ticket)

        return {
            "query": query.query,
            "tickets":tickets
            }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post('/api/generate-email-draft')
async def generate_email_draft(similarTickets: dict):
    try:
        query = similarTickets['query']
        tickets = similarTickets['tickets']
        answers = [ticket['answer'] for ticket in tickets]

        draft = [ticket['answer'] for ticket in tickets]

        context = '\n'.join(draft)

        prompt = f"""
        Compose a support email in a friendly tone addressing the provided query and using only the relevant contexts. Adhere to the following guidelines:

        1. Begin the email with a friendly greeting.
        2. Must keep the email not more than 200 tokens.
        3. Must present process steps using bullet points.
        4. Only use context information essential to address the query; avoid unnecessary details.
        5. Incorporate placeholders where needed.
        6. Provide the essential information requested without including unsolicited details.

        Query:
        {query}

        Contexts:
        {context}
        """
    
        email_draft = ask(prompt)
        
        print(email_draft)

        return {"email_draft": email_draft}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")
    

@app.post('/api/generate-email-draft-v2')
async def generate_email_draft(similarTickets: dict):

    try:
        query = similarTickets['query']
        case_id = similarTickets['caseId']
        case_number = similarTickets['caseNumber']
        contexts = [f"Question: {con['question']}\nAnswer:\n{con['answer']}" for con in similarTickets['relatedCases']]

        most_related_contexts = get_similar_score(query, contexts, relatedness_fn)

        # print(most_related_contexts)

        if len(most_related_contexts) == 0:
            email_draft = "Sorry, I couldn't find any relevant information to address your query."
            
        else:
        
            context = '\n'.join(most_related_contexts)

            prompt = f"""
            Compose a support email in a friendly tone addressing the provided query and relevant contexts. Adhere to the following guidelines:
            
            1. Begin the email with a friendly greeting.
            2. Keep the email concise and within 200 tokens.
            3. Use bullet points to present process steps.
            4. Include only essential context information to address the query.
            5. Incorporate placeholders where needed.
            6. Provide the requested information without unnecessary details.
            7. Strictly use the information in the provided context.

            Query:
            {query}

            Contexts:
            {context}
            """

            print(prompt)

            email_draft = ask(prompt)
            # email_draft = prompt
        
        print(email_draft)

        return {
            "case_id": case_id,
            "case_number": case_number,
            "email_draft": [email_draft]
            }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")
