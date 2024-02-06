// scripts.js
async function getSimilarTickets() {
    const query = document.getElementById('queryTextarea').value;

    const loadingSpinner = document.getElementById('loadingSpinner');
    loadingSpinner.style.display = 'block';

    try{
        const response = await fetch('http://localhost:3000/api/similar-tickets', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ "query":query }),
        });
    
        const data = await response.json();
        
        document.getElementById('resultTextarea').value = JSON.stringify(data, null, 2);
    }
    catch(error) {
        console.log(error)
    }
    finally {
        // Hide loading spinner after fetch is complete
        loadingSpinner.style.display = 'none';
    }
    
}

async function generateEmailDraft() {
    const query = document.getElementById('resultTextarea').value;

    // console.log(query)
    const loadingSpinner = document.getElementById('loadingSpinner');
    loadingSpinner.style.display = 'block';

    try {
        const response = await fetch('http://localhost:3000/api/generate-email-draft', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: query,
    });

    const data = await response.json();
    document.getElementById('emailTextarea').value = JSON.stringify(data, null, 2);
    }

    catch(error){
        console.log(error)
    }
    finally{
        loadingSpinner.style.display = 'none';
    }
    
}
