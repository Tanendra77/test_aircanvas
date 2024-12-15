st.title("Magpasa ng papel pang obserba ng sintomas")
st.write('1. Piliin ang araw NA NAKASULAT SA PAPEL ng pasyente.')
date = st.date_input('Araw ng pag record')
st.write('2. Iclick ang allow sa pag gamit ng camera at picturan ang buong papel o mag upload ng larawan ng papel.')
st.write('3. Intayin ang kumpirmasyon na naupload ang resulta')
st.title("")
st.title("")
cam = st.camera_input(label='Kunan ng letrato ang papel',disabled=False)
file = st.file_uploader('O mag upload ng larawan ng papel', type=["png", "jpg", "jpeg"])
image_main = None
if cam is not None:
    image_main = cam
elif file is not None:
    image_main = file
if image_main is not None:
    img = Image.open(image_main)
    with st.spinner('Submission in progress'):
        img.save("./inputs/test.jpg")
        run_analyzer('test.jpg')
