step 1 : Go to regrest.py and install the required libraries , I have mentioned it in requirements.txt but if still something is missing , you can install it by using pip install

step 2: Flask is used here to host it , when you will run it , it will create a link for you you have to open that link and it is just to host our rag model , it is not where you will ask questions .
(You can use postman to verify the status)

step 3: open rag_frontend.html file and either you can open it on compiler or directly open it , you will see the rag chatbot working perfectly , 
NOTE: this html file will only work if you have gone through step 1 and 2 as they are crucial to host our rag model 

IMPORTANT POINTS :
** I have used paid Azure API here , if you have it then ok otherwise you can go for free ones line OLLAma 
** create a .env file and save your ap credentials there .
** Huggingface Embeddings have been used here
