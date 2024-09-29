"use client"
import DragDrop from "./DragDrop";
import axios from "axios";
import React, {useEffect, useState} from "react";



export default function Home() {

  const [image, setImage] = useState(`data:image/png;base64,`);
  const [update, setUpdate] = useState(false);
  const [visibility, setVis] = useState(false);
  const [cancer, setCancer] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      if(!update) {
        return ;
      }
      const config = {
        headers: {
          'Accept': 'application/json'
        }
      };
      try {
        const res1 = await axios.get("http://localhost:8000/api/image/fetch-image", config).then(update?setVis(true):pass);
        const res2 = await axios.get("http://localhost:8000/api/image/fetch-result", config).then(update?setVis(true):pass);
        console.log(res1.data);
        console.log(res1);
        if( res1.status == 200) {
          setCancer(res2.data);
          setImage(`data:image/png;base64, ${res1.data}`);
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
          <h1 className="bg-gray-100 p-10 rounded-xl">{cancer? "This cell has a malignant tumor": "This cell has no malignant tumor."}</h1>
        <img width="500" height="500"  src={image} key={Date.now()} alt="temp" />
        <button className="mt-10 w-40 h-10 bg-gray-100 flex justify-center items-center rounded-xl bold" onClick={reset}>Reset</button>
      </div>
    </div>
  );
}
