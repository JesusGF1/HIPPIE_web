import os
os.environ["PYNWB_NO_CACHE_DIR"] = "1"          #disable typemap caching
os.environ.setdefault("PYNWB_CACHE_DIR", "/tmp/pynwb_cache")  #just in case

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import tempfile
from sklearn.neighbors import KNeighborsClassifier
from bokeh.models import ColumnDataSource
import tarfile
from neurocurator import Neurocurator

from utils import normalize_to_minus1_1, normalize_by_row_max, plotter, compute_umap, csv_downloader, compute_pumap, HIPPIE, compue_the_clusters_kmeans, load_data_classifier, compue_the_clusters_labeled, compue_the_clusters_hdbscan, resize_rows_linear, acqm_file_reader_np, download_drive_file



st.set_page_config(layout="wide", page_title="Neural data visualizer", page_icon=":bar_chart:")

st.markdown("""
<style>

/* FULL APP BACKGROUND */
.stApp, html, body, [class*="css"] {
    background-color: #000000 !important;
}

/* Remove the default white background on inner containers */
.block-container {
    background-color: transparent !important;
}

/* Sidebar full background */
[data-testid="stSidebar"] > div:first-child {
    background-color: #000000 !important;
}

/* Make widget backgrounds transparent so black shows through */
.stTextInput>div>div>input,
[data-baseweb="select"] > div,
.stFileUploader,
.stDateInput,
div[data-baseweb="textarea"] textarea {
    background-color: transparent !important;
    color: white !important;
}

/* Button backgrounds transparent (unless you recolor them separately) */
.stButton>button {
    background-color: transparent !important;
    border-color: white !important;
    color: white !important;
}

/* Scrollbars dark as well */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: #444;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Force white text across the whole app */
html, body, .stApp {
    color: #FFFFFF !important;
}

/* Markdown text */
[data-testid="stMarkdownContainer"] {
    color: #FFFFFF !important;
}

/* Generic divs (most Streamlit text sits in these) */
div, p, span, label {
    color: #FFFFFF !important;
}

/* Text inside inputs */
input, textarea {
    color: #FFFFFF !important;
}

/* Prevent browser from auto-inverting text based on OS theme */
html {
    color-scheme: dark;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Style ONLY Streamlit's download button */
.stDownloadButton > button {
    background-color: #000000 !important;  /* pure black */
    color: #FFFFFF !important;             /* white text */
    border: 1px solid #FFFFFF !important;  /* optional white border */
}

/* On hover */
.stDownloadButton > button:hover {
    background-color: #222222 !important;  /* slightly lighter black */
    color: #FFFFFF !important;
}

/* Prevent OS dark/light mode overrides */
.stDownloadButton > button {
    -webkit-appearance: none !important;
    appearance: none !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* ==== DARK FILE UPLOADER DROPZONE ==== */

/* Outer uploader container */
[data-testid="stFileUploader"] {
    background-color: #000000 !important;
    border-radius: 6px !important;
}

/* The big drag-and-drop area (the white bar in your screenshot) */
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section > div,
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploadDropzone"] {
    background-color: #000000 !important;
    border: 1px dashed #444444 !important;
    border-radius: 6px !important;
}

/* Text inside dropzone ("Drag and drop files here", etc.) */
[data-testid="stFileUploader"] section * {
    color: #FFFFFF !important;
}

/* "Browse files" button inside uploader */
[data-testid="stFileUploaderBrowseButton"] {
    background-color: #000000 !important;
    color: #FFFFFF !important;
    border: 1px solid #FFFFFF !important;
}
[data-testid="stFileUploaderBrowseButton"]:hover {
    background-color: #222222 !important;
}

/* Uploaded file row (the line with the filename) */
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    background-color: #000000 !important;
    color: #FFFFFF !important;
}

/* File name + details text */
[data-testid="stFileUploaderFileName"],
[data-testid="stFileUploaderFileDetails"] {
    color: #FFFFFF !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* ==== FORCE BROWSE BUTTON TO BLACK ==== */

/* Primary test-id (newer Streamlit DOM) */
[data-testid="stFileUploadDropzone"] button,
[data-testid="stFileUploader"] button {

/* Dark style */
    background-color: #000000 !important;
    color: #FFFFFF !important;
    border: 1px solid #FFFFFF !important;
    border-radius: 6px !important;
}

/* Hover state */
[data-testid="stFileUploadDropzone"] button:hover,
[data-testid="stFileUploader"] button:hover {
    background-color: #222222 !important;
    color: #FFFFFF !important;
}

/* Reset browser visual tweaking */
[data-testid="stFileUploadDropzone"] button,
[data-testid="stFileUploader"] button {
    -webkit-appearance: none !important;
    appearance: none !important;
}

</style>
""", unsafe_allow_html=True)


st.title("Neural data visualizer")
st.write("Upload your CSV data files and visualize them please")



#load the curated dataset
#acg_a, isi_a, wf_a, ct_a = load_data_classifier('classifying_data.tar.xz')

uploaded_file_cell_type = None
#with st.container():
#    uploaded_file_cell_type = st.file_uploader("upload the cell_type .csv file if you have one")
#    if uploaded_file_cell_type is not None:
#
#        # Can be used wherever a "file-like" object is accepted:
#        dataframe_cell_type = pd.read_csv(uploaded_file_cell_type).astype(str)
#        #st.write(dataframe_isi_dist)
#
#        #dataframe_cell_type.iloc[:, 0] = dataframe_cell_type.iloc[:, 0].apply(lambda x: x.decode() if isinstance(x, bytes) else x)


uploaded_acg_files = []
uploaded_isi_files = []
uploaded_waveform_files = []
uploaded_phisiological_data_zip = []
nwb_uploaded = []
phy_uploaded = []
token_csv = False
token_acqm = False
token_nwb = False
token_phy = False
token_link = False

uploading_option = st.radio(
    "Choose input method:",
    ("Work with csv files", "Work with acqm.zip files", "Work with nwb files", "Work with phy files", 'work with download link')
)

if uploading_option == "Work with csv files":
    col1, col2, col3 = st.columns(3)
    with col1:
        uploaded_acg_files = st.file_uploader(
            "Upload the acg .csv files here", accept_multiple_files=True, type=["csv"]
        ) or []
    with col2:
        uploaded_isi_files = st.file_uploader(
            "Upload the isi .csv files here", accept_multiple_files=True, type=["csv"]
        ) or []
    with col3:
        uploaded_waveform_files = st.file_uploader(
            "Upload the waveform .csv files here", accept_multiple_files=True, type=["csv"]
        ) or []

    # tokens become True only when lists are non-empty
    if len(uploaded_acg_files) > 0 and len(uploaded_isi_files) > 0 and len(uploaded_waveform_files) > 0:
        token_csv = True

    # make sure the other inputs are clearly "unused"
    uploaded_phisiological_data_zip = []
    nwb_uploaded = []

elif uploading_option == "Work with acqm.zip files":
    uploaded_phisiological_data_zip = st.file_uploader(
        "Upload your acqm.zip files here", accept_multiple_files=True, type=["zip"]
    ) or []

    if len(uploaded_phisiological_data_zip) > 0:
        token_acqm = True

    uploaded_acg_files = []
    uploaded_isi_files = []
    uploaded_waveform_files = []
    nwb_uploaded = []

elif uploading_option == "Work with nwb files":
    nwb_uploaded = st.file_uploader(
        "Upload your nwb files here", type=["nwb"], accept_multiple_files=True
    ) or []

    if len(nwb_uploaded) > 0:
        token_nwb = True

    uploaded_phisiological_data_zip = []
    uploaded_acg_files = []
    uploaded_isi_files = []
    uploaded_waveform_files = []

elif uploading_option == "Work with phy files":
    phy_uploaded = st.file_uploader(
        "Upload your phy files here", type=["zip"], accept_multiple_files=True
    ) or []

    if len(phy_uploaded) > 0:
        token_phy = True

    uploaded_phisiological_data_zip = []
    uploaded_acg_files = []
    uploaded_isi_files = []
    uploaded_waveform_files = []

elif uploading_option == "work with download link":
    url = st.text_input("Paste a Google Drive direct link (share link is fine):")

    file_kind = st.selectbox("What are you linking to?", ["acqm.zip", "nwb", "phy.zip"])
    
    if len(url) > 0:
        token_link = True

if token_acqm or token_csv or token_nwb or token_phy or token_link:


    if token_acqm:
        acg_parts = []
        isi_parts = []
        wf_parts  = []

        for uploaded_file in uploaded_phisiological_data_zip:
            
            #write the uploaded zip to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name

            try:
                acg_np, isi_np, wf_np = acqm_file_reader_np(tmp_path)

                #store the results without concatenating yet
                acg_parts.append(acg_np)
                isi_parts.append(isi_np)
                wf_parts.append(wf_np)

            finally:

                #clean up the temp file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            #free references of each data early on
            del acg_np, isi_np, wf_np

        #one big stack
        acg_all = np.vstack(acg_parts) if acg_parts else np.empty((0, 100), dtype=np.float32)
        isi_all = np.vstack(isi_parts) if isi_parts else np.empty((0, 100), dtype=np.float32)
        wf_all  = np.vstack(wf_parts)  if wf_parts  else np.empty((0, 50),  dtype=np.float32)

        #convert back to DataFrame once (normalize/plotter code stays identical)
        df_acg = pd.DataFrame(acg_all)
        df_isi = pd.DataFrame(isi_all)
        df_waveforms = pd.DataFrame(wf_all)

        #free the parts lists for RAM
        del acg_parts, isi_parts, wf_parts, acg_all, isi_all, wf_all
        import gc; gc.collect()


    elif token_csv:
        acg_np = []
        isi_np = []
        wf_np  = []

        for f in uploaded_acg_files:
            acg_np.append(pd.read_csv(f).to_numpy(dtype=np.float32))

        for f in uploaded_isi_files:
            isi_np.append(pd.read_csv(f).to_numpy(dtype=np.float32))

        for f in uploaded_waveform_files:
            wf_np.append(pd.read_csv(f).to_numpy(dtype=np.float32))

        acg_all = np.vstack(acg_np) if acg_np else np.empty((0, 100), dtype=np.float32)
        isi_all = np.vstack(isi_np) if isi_np else np.empty((0, 100), dtype=np.float32)
        wf_all  = np.vstack(wf_np)  if wf_np  else np.empty((0, 50),  dtype=np.float32)

        df_acg = pd.DataFrame(acg_all)
        df_isi = pd.DataFrame(isi_all)
        df_waveforms = pd.DataFrame(wf_all)

        del acg_np, isi_np, wf_np, acg_all, isi_all, wf_all
        import gc; gc.collect()



    elif token_nwb:
        acg_np = []
        isi_np = []
        wf_np  = []

        for uploaded_file in nwb_uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".nwb") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name

            try:
                nc = Neurocurator()
                nc.load_nwb_spike_times(tmp_path)
                nc.load_nwb_waveforms(tmp_path, n_datapoints=50,
                                    candidates=("waveform_mean", "spike_waveforms"))

                nc.isi_distribution = nc.compute_isi_distribution(time_window=100)
                nc.acgs = nc.compute_autocorrelogram(nc.spike_times_train)

                acg_np.append(nc.acgs.to_numpy(dtype=np.float32, copy=True))
                isi_np.append(nc.isi_distribution.to_numpy(dtype=np.float32, copy=True))
                wf_np.append(nc.waveforms.to_numpy(dtype=np.float32, copy=True))

            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            del nc

        acg_all = np.vstack(acg_np) if acg_np else np.empty((0, 100), dtype=np.float32)
        isi_all = np.vstack(isi_np) if isi_np else np.empty((0, 100), dtype=np.float32)
        wf_all  = np.vstack(wf_np)  if wf_np  else np.empty((0, 50),  dtype=np.float32)

        df_acg = pd.DataFrame(acg_all)
        df_isi = pd.DataFrame(isi_all)
        df_waveforms = pd.DataFrame(wf_all)

        del acg_np, isi_np, wf_np, acg_all, isi_all, wf_all
        import gc; gc.collect()

    
    elif token_phy:
        acg_np = []
        isi_np = []
        wf_np  = []

        for uploaded_file in phy_uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name

            try:
                nc = Neurocurator()
                nc.load_phy_curated(tmp_path)

                acg_np.append(nc.acgs.to_numpy(dtype=np.float32, copy=True))
                isi_np.append(nc.isi_distribution.to_numpy(dtype=np.float32, copy=True))
                wf_np.append(nc.waveforms.to_numpy(dtype=np.float32, copy=True))

            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            del nc

        acg_all = np.vstack(acg_np) if acg_np else np.empty((0, 100), dtype=np.float32)
        isi_all = np.vstack(isi_np) if isi_np else np.empty((0, 100), dtype=np.float32)
        wf_all  = np.vstack(wf_np)  if wf_np  else np.empty((0, 50),  dtype=np.float32)

        df_acg = pd.DataFrame(acg_all, columns=[f"acg_{i}" for i in range(acg_all.shape[1])])
        df_isi = pd.DataFrame(isi_all, columns=[f"isi_{i}" for i in range(isi_all.shape[1])])
        df_waveforms = pd.DataFrame(wf_all, columns=[f"wf_{i}" for i in range(wf_all.shape[1])])

        del acg_np, isi_np, wf_np, acg_all, isi_all, wf_all
        import gc; gc.collect()
    
    elif token_link:
        
        suffix = ".zip" if "zip" in file_kind else ".nwb"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
        
        try:
            #to prevent from downloading the same file again and again with each iteratins, since streamlit re runs everything each time it access utils or the user touches something aparently
            from utils import _gdrive_download_url, _gdrive_file_id
            import hashlib
            suffix = ".zip" if "zip" in file_kind else ".nwb"

            # Make a deterministic cache path for ALL links
            h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
            cache_path = f"/tmp/{h}{suffix}"

            # Normalize Drive links if possible
            fid = _gdrive_file_id(url)
            if fid:
                url = _gdrive_download_url(fid)

            # Download only if not cached
            if (not os.path.exists(cache_path)) or (os.path.getsize(cache_path) < 1024):
                download_drive_file(url, cache_path)

            tmp_path = cache_path

            if file_kind=="acqm.zip":
                acg_np, isi_np, wf_np = acqm_file_reader_np(tmp_path)
                df_acg = pd.DataFrame(acg_np)
                df_isi = pd.DataFrame(isi_np)
                df_waveforms = pd.DataFrame(wf_np)
            
            elif file_kind=="nwb":
                nc = Neurocurator()
                nc.load_nwb_spike_times(tmp_path)
                nc.load_nwb_waveforms(tmp_path, n_datapoints=50, candidates=("waveform_mean", "spike_waveforms"))

                nc.isi_distribution = nc.compute_isi_distribution(time_window=100)
                nc.acgs = nc.compute_autocorrelogram(nc.spike_times_train)
                df_acg = pd.DataFrame(nc.acgs.to_numpy(dtype=np.float32, copy=False))
                df_isi = pd.DataFrame(nc.isi_distribution.to_numpy(dtype=np.float32, copy=False))
                df_waveforms = pd.DataFrame(nc.waveforms.to_numpy(dtype=np.float32, copy=False))

            
            elif file_kind=='phy.zip':
                nc = Neurocurator()
                nc.load_phy_curated(tmp_path)
                df_acg = pd.DataFrame(nc.acgs.to_numpy(dtype=np.float32, copy=False))
                df_isi = pd.DataFrame(nc.isi_distribution.to_numpy(dtype=np.float32, copy=False))
                df_waveforms = pd.DataFrame(nc.waveforms.to_numpy(dtype=np.float32, copy=False))

        
        finally:
            pass


#print("########################FILES LOADED#############################")
#################################




###################################
    col1, col2, col3 = st.columns(3)
    with col1:
    #acg file
        resized_acg = resize_rows_linear(df_acg.values, 100)
        normalized_acg = normalize_to_minus1_1(df_acg)
        p = plotter(normalized_acg, 'ACG', 'Timepoint', 'Amplitude')
        st.bokeh_chart(p, use_container_width=True)

    with col2:
    #isi plot
        resized_isi = resize_rows_linear(df_isi.values, 100)
        normalized_isi = normalize_by_row_max(df_isi)
        p = plotter(normalized_isi, 'ISI distribution', 'Timepoint', 'Amplitude')
        st.bokeh_chart(p, use_container_width=True)
    
    with col3:
    #waveforms plot
        resized_waveforms = resize_rows_linear(df_waveforms.values, 50)
        normalized_waveforms = normalize_to_minus1_1(df_waveforms)
        p = plotter(normalized_waveforms, 'Waveforms', 'Timepoint', 'Amplitude')
        st.bokeh_chart(p, use_container_width=True)
    

        
    #resized_acg_a = F.interpolate(
    #    torch.tensor(acg_a.values, dtype=torch.float32).unsqueeze(1),
    #    size=100,
    #    mode='linear'
    #).squeeze(1).numpy()
    #        
    #resized_isi_a = F.interpolate(
    #            torch.tensor(isi_a.values, dtype=torch.float32).unsqueeze(1),
    #            size=100,
    #            mode='linear'
    #        ).squeeze(1).numpy()
    #    
    #resized_wf_a = F.interpolate(
    #    torch.tensor(wf_a.values, dtype=torch.float32).unsqueeze(1),
    #    size=50,
    #    mode='linear'
    #).squeeze(1).numpy()

    # Technology conditioning for HIPPIE encoder (matches training vocabulary)
    TECHNOLOGY_OPTIONS = {
        "Neuropixels (1.0 / 2.0)":           0,  # IBL, Allen, Hausser, Hull, CellExplorer
        "Silicon probe (non-Neuropixels)":    1,  # NeuroNexus, Plexon s-Probe, etc.
        "Juxtacellular (glass micropipette)": 2,
        "Tetrode":                            3,
    }

    source = TECHNOLOGY_OPTIONS[st.selectbox(
        "Select recording technology",
        options=list(TECHNOLOGY_OPTIONS.keys()),
        help="Choose the electrode type used to record your data. "
             "This conditions the HIPPIE encoder on recording technology.",
    )]

    #get HIPPIE embedings

    #THIS USED TO JOIN THE UNDERNEATH DATASET TO ADD CELLTYPES WHEN NO FILE PROVIDED
    #acg_T = pd.concat([pd.DataFrame(resized_acg_a), pd.DataFrame(resized_acg)], ignore_index=True)
    #isi_T = pd.concat([pd.DataFrame(resized_isi_a), pd.DataFrame(resized_isi)], ignore_index=True)
    #wf_T = pd.concat([pd.DataFrame(resized_wf_a), pd.DataFrame(resized_waveforms)], ignore_index=True)
    acg_T = resized_acg
    isi_T = resized_isi
    wf_T = resized_waveforms


    #create a multimodal dataset with all modalities
    #also make it numpy arrays because MultiModalEphysDataset expects numPy arrays
    embedding, labels = HIPPIE(pd.DataFrame(acg_T), pd.DataFrame(isi_T), pd.DataFrame(wf_T), source)
    
    ##################################################
    #PUMAP
    #loading the onnx model
    #output_array = compute_pumap(embedding)

    #x = list(range(len(output_array)))
    #y = output_array[:, 0]

    #source = ColumnDataSource(data=dict(x=x, y=y))
    #output_array = pd.DataFrame(output_array, columns=['UMAP 1', 'UMAP 2'])
    ################################################


    ##########################################################################
    #normal UMAP

    #if no valid neurons survived cleaning/normalization the code will stop
    if embedding is None or len(embedding) == 0:
        st.error(
            "No valid neurons remained after preprocessing. "
            "This often happens if all units have flat features (e.g. no spikes or "
            "no waveform/ISI variance) in the NWB file."
        )
        st.stop()



    embedding = compute_umap(embedding)
    output_array = np.array(embedding, dtype=np.float32)
    output_array = pd.DataFrame(output_array, columns=['UMAP 1', 'UMAP 2'])
    ##########################################################################

    if uploaded_file_cell_type is None:
        
        min_cluster_size = st.slider(
            "Minimum cluster size",
            min_value=5,
            max_value=200,
            value=15,
            help="Smallest allowed cluster size. Higher → fewer, more stable clusters."
        )

        min_samples = st.slider(
            "Minimum samples",
            min_value=5,
            max_value=200,
            value=15,
            help="Controls density strictness. Higher → only very dense clusters survive."
        )


        #computing the hdbscan clustering
        output_array = compue_the_clusters_hdbscan(output_array, min_cluster_size, min_samples)

        #making the chart
        chart = alt.Chart(output_array).mark_circle(size=30).encode(
            x='UMAP 1',
            y='UMAP 2',
            color='Classifier:N',
        ).properties(
            width=800,
            height=800,
            background='#000000'
        )
        
        st.altair_chart(chart, use_container_width=True)


        #here goes the choosing clusters thing
        option = st.selectbox(
            'Select a cluster to see the corresponding data points',
            options=output_array['Classifier'].unique()
        )
        st.write(f"You selected cluster {option}")

        selected = output_array[output_array['Classifier'] == option]
        acg_types = pd.concat([normalized_acg, output_array['Classifier']], axis=1)
        isi_types = pd.concat([normalized_isi, output_array['Classifier']], axis=1)
        waveforms_types = pd.concat([normalized_waveforms, output_array['Classifier']], axis=1)

        sel_acg_types = acg_types[acg_types['Classifier'] == option]
        sel_isi_types = isi_types[isi_types['Classifier'] == option]
        sel_waveforms_types = waveforms_types[waveforms_types['Classifier'] == option]


        ############################################

        #########Downloading files box##############
        #downloading_option = st.radio(
        #    "Choose a downloading option:",
        #    ("Download .csv (multiple files)", "Download .h5 (1 file)")
        #)
        #if downloading_option == "Download .h5 (1 file)":
        #    h5_downloader(output_array, acg_types, isi_types, waveforms_types)
        #elif downloading_option == "Download .csv (multiple files)":
        csv_downloader(output_array, acg_types, isi_types, waveforms_types)
            
        ############################################
        
        p = plotter(acg_types, 'ACG with clusters', 'Timepoint', 'Amplitude', option)
        st.bokeh_chart(p, use_container_width=True)
        
        p = plotter(isi_types, 'ISI distribution with clusters', 'Timepoint', 'Amplitude', option)
        st.bokeh_chart(p, use_container_width=True)
        
        p = plotter(waveforms_types, 'Waveforms with clusters', 'Timepoint', 'Amplitude', option)
        st.bokeh_chart(p, use_container_width=True)
    
    elif uploaded_file_cell_type is not None:
        
        st.title('Parametric UMAP with cell type information')
        output_array['Cluster'] = dataframe_cell_type

        #making the chart
        chart = alt.Chart(output_array).mark_circle(size=30).encode(
            x='UMAP 1',
            y='UMAP 2',
            color='Cluster',
        ).properties(
            width=800,
            height=800
        )
        st.altair_chart(chart, use_container_width=True)

        # Dropdown to select a type
        option = st.selectbox(
            'Select a cell type to see the corresponding data points',
            options=output_array['Cluster'].unique()
        )

        # Filter output_array and apply to each data modality
        selected = output_array[output_array['Cluster'] == option]
        acg_types = pd.concat([normalized_acg, output_array['Cluster']], axis=1)
        isi_types = pd.concat([normalized_isi, output_array['Cluster']], axis=1)
        waveforms_types = pd.concat([normalized_waveforms, output_array['Cluster']], axis=1)

        sel_acg_types = acg_types[acg_types['Cluster'] == option]
        sel_isi_types = isi_types[isi_types['Cluster'] == option]
        sel_waveforms_types = waveforms_types[waveforms_types['Cluster'] == option]

        # Download
        csv_downloader(output_array, acg_types, isi_types, waveforms_types)

        # Plotting
        p = plotter(acg_types, 'ACG with celltypes', 'Timepoint', 'Amplitude', option)
        st.bokeh_chart(p, use_container_width=True)

        p = plotter(isi_types, 'ISI with celltypes', 'Timepoint', 'Amplitude', option)
        st.bokeh_chart(p, use_container_width=True)

        p = plotter(waveforms_types, 'Waveforms with celltypes', 'Timepoint', 'Amplitude', option)
        st.bokeh_chart(p, use_container_width=True)

    

    #Ploting only with means per cluster 

    st.title('Ploting only with means per cluster')

    acg_mean_list = []
    isi_mean_list = []
    wf_mean_list = []
    cluster_labels = []

    if uploaded_file_cell_type is None:
        for label in acg_types['Classifier'].unique():
            acg_mean_list.append(np.mean(acg_types[acg_types['Classifier'] == label].drop(columns='Classifier'), axis=0))
            isi_mean_list.append(np.mean(isi_types[isi_types['Classifier'] == label].drop(columns='Classifier'), axis=0))
            wf_mean_list.append(np.mean(waveforms_types[waveforms_types['Classifier'] == label].drop(columns='Classifier'), axis=0))
            cluster_labels.append(label)


    elif uploaded_file_cell_type is not None:
        for x in output_array['Classifier'].unique():
            acg_mean_list.append(np.mean(acg_types[acg_types['Classifier']==x].drop(columns='Classifier'), axis=0))
            isi_mean_list.append(np.mean(isi_types[isi_types['Classifier']==x].drop(columns='Classifier'), axis=0))
            wf_mean_list.append(np.mean(waveforms_types[waveforms_types['Classifier']==x].drop(columns='Classifier'), axis=0))

            cluster_labels.append(x)
        
    acg_mean_list = pd.DataFrame(acg_mean_list)
    isi_mean_list = pd.DataFrame(isi_mean_list)
    wf_mean_list = pd.DataFrame(wf_mean_list)

    acg_mean_list['Classifier'] = cluster_labels
    isi_mean_list['Classifier'] = cluster_labels
    wf_mean_list['Classifier'] = cluster_labels



    p = plotter(pd.DataFrame(acg_mean_list), 'ACG_mean', 'Timepoint', 'Amplitude', selected_cluster=option, alpha_background=0.8, alpha_upfront=0.8, line_width_background=0.8, line_width_upfront=3)
    st.bokeh_chart(p, use_container_width=True)
    p=plotter(pd.DataFrame(isi_mean_list), 'isi_mean', 'Timepoint', 'Amplitude', selected_cluster=option, alpha_background=0.8, alpha_upfront=0.8, line_width_background=0.8, line_width_upfront=3)
    st.bokeh_chart(p, use_container_width=True)
    p=plotter(pd.DataFrame(wf_mean_list), 'Waveforms_mean', 'Timepoint', 'Amplitude', selected_cluster=option, alpha_background=0.8, alpha_upfront=0.8, line_width_background=0.8, line_width_upfront=3)
    st.bokeh_chart(p, use_container_width=True)

    

else:
    st.info("Please upload all required files (ACG, ISI distribution, and waveforms) to create the UMAP visualization.")