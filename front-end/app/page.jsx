"use client"
import DragDrop from "./DragDrop";
import axios from "axios";
import React, {useEffect, useState} from "react";



export default function Home() {

  const [image, setImage] = useState(`data:image/png;base64,`);
  const [update, setUpdate] = useState(false);
  const [visibility, setVis] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      const config = {
        headers: {
          'Accept': 'application/json'
        }
      };
      try {
        const res = await axios.get("http://localhost:8000/api/image/fetch-image", config).then(update?setVis(true):pass);
        console.log(res.data);
        console.log(res);
        if( res.status == 200) {
          setImage(`data:image/png;base64, ${res.data}`);
        }
      } catch(err) {
        console.log(err)
      }
    };

    fetchData();
  }, [update])

  const onClick = e => {
    e.preventDefault();
    setUpdate(!update);
    setImage("load.gif");
  };

  const reset = e => {
    e.preventDefault();
    setVis(false);
  }

  return (
    <div className="flex justify-center items-center w-screen h-screen">
      <form className="flex flex-col justify-center items-center h-full" style={{"position": "relative", "left":"13em", "flex": "column", "visibility": visibility ? "hidden":"visible" }} encType="multipart/form-data">
        <DragDrop />
        <button className="w-40 h-10 bg-gray-100 flex justify-center items-center rounded-xl bold m-10" onClick={onClick}>Get Result</button>
      </form>
        <div className="flex flex-col justify-center items-center content-center" style={{"position": "relative" ,"right": "12em", "visibility": visibility ? "visible":"hidden"}}>
        <img width="500" height="500"  src={image} key={Date.now()} alt="temp" />
        <button className="mt-10 w-40 h-10 bg-gray-100 flex justify-center items-center rounded-xl bold" onClick={reset}>Reset</button>
      </div>
    </div>
  );
}
