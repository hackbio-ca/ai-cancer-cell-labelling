"use client"

import {FileUploader} from "react-drag-drop-files";


function DragDrop(onChange) {
    return ( 
        <FileUploader multiple handleChange={handleChange} name="file" onChange={onChange}/>
    )
}

export default DragDrop;