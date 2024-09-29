"use client"

import {FileUploader} from "react-drag-drop-files";
import axios from 'axios'


function DragDrop() {

    const handleChange = async (e) => {


        const formData = new FormData();
        console.log(e)
        formData.append("image", e)
        const body = formData;
        const config = {
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'multipart/form-data',
                }
            };

        try {
            const res = await axios.post("http://localhost:8000/api/image/upload",body, config);
            if(res.status == 200) {
                return;
            }
        } catch(err) {  
        console.log(err);
        }
    }

    return ( 
        <FileUploader handleChange={handleChange} name="file" />
    )
}

export default DragDrop;