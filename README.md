# support-helper-api

## Must Navigate to the Server Directory
```bash
cd server/
```


## Setting Up a Virtual Environment

1. Install the virtual environment package if it's not already installed:

    ```bash
    pip install virtualenv
    ```

2. Create a new virtual environment in your project directory:

    ```bash
    virtualenv venv
    ```

3. Activate the virtual environment:

- On Windows:

    ```bash
    venv\Scripts\activate
    ```

- On Unix or MacOS:

    ```bash
    source venv/bin/activate
    ```

## Installing Dependencies

- After activating the virtual environment, install the project dependencies with:

    ```bash
    pip install -r requirements.txt
    ```

## Creating a `.env` File

1. In the root of your project directory, create a new file named `.env`.

2. Open the `.env` file and add your OpenAI API key like this:

    ```bash
    OPENAI_API_KEY=your-api-key-here
    ```
    *Replace `your-api-key-here` with your actual OpenAI API key.*

3. Save and close the `.env` file.

## Running the Server

- To run the server, use the following command:

    ```bash
    uvicorn server:app --reload --host 0.0.0.0 --port 3000
    ```