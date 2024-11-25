import streamlit as st
import utils, indexing

st.set_page_config("Book Analyzer", page_icon=":books:")

st.title("Book Analyzer")

file = st.file_uploader("Upload novel PDF", type=['pdf'])
if file is not None:
    if 'index' not in st.session_state:
        with st.spinner("Making index... (this might take a while)"):
            indexing.create_index(file)
            index, metadata = indexing.load_index()
            st.session_state['index'] = index
            st.session_state['metadata'] = metadata

    query = st.text_input("Query")
    if query:
        chunks = indexing.query_index(query, st.session_state['index'], st.session_state['metadata'])
        print(chunks)
        prompt = utils.get_prompt(query, chunks)
        st.text(utils.get_response(prompt))

        st.subheader("Quotes from the novel")
        for chunk in chunks:
            st.write(f"{chunk['text']} ({(chunk['page_number'])})\n")
