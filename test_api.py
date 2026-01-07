import requests
import streamlit as st

st.title("🔍 API Debug")

# PUBLIC OMDb KEY
OMDB_API_KEY = " f5127ade"

if st.button("🖼️ Test OMDb Avatar"):
    url = "http://www.omdbapi.com/"
    params = {"apikey": OMDB_API_KEY, "t": "Avatar", "r": "json"}
    
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        
        st.json(data)  # Full response
        
        if data.get("Response") == "True":
            poster = data.get("Poster")
            st.success(f"✅ POSTER: {poster}")
            if poster and poster != "N/A":
                st.image(poster)
        else:
            st.error(f"❌ Error: {data.get('Error')}")
            
    except Exception as e:
        st.error(f"Network: {e}")

st.info("✅ Shows JSON + poster = API good\n❌ Error/blank = network/proxy issue")
