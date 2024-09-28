import Image from "next/image";
import DragDrop from "./DragDrop";
import axios from "axios";
import {useState, useEffect} from "react";

export default function Home() {

  const onSubmit = async (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();

    const config = {
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'multipart/form-data',
      }
    };

    console.log(e.target.files);
    const formData = new FormData();
    formData.append("images", e.target.files);
    const body = formData;

    try {
      const res = await axios.post("http://localhost:8000/api/image/upload", body, config);
      if(res.status == 200) {
        return;
      }
    } catch(err) {  
      console.log(err);
    }
  }

  return (
    <div className="flex justify-center items-center w-screen h-screen">
      <form className="flex justify-center items-center h-full" style={{"flex-direction":"column"}}>
          {/* <label htmlFor="file-upload" className="w-80 h-60 bg-gray-100 flex justify-center items-center rounded-xl bold"> Please drag files or click to select files.</label>
          <input id="file-upload" type="file" multiple style={{"opacity": 0}}/> */}
          <DragDrop onChange={onSubmit}/>
          <button className="w-40 h-10 bg-gray-100 flex justify-center items-center rounded-xl bold" style={{"margin":"10px"}}>Get Result</button>
      </form>
    </div>
  );
}
