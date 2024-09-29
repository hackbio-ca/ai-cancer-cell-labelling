
import DragDrop from "./DragDrop";
import Image from "next/image";
import { useState, useEffect } from 'react';
import axios from "axios";



export default function Home() {

  const [image, setImage] = useState(null);
  const [updated, setUpdated] = useState(false);

  useEffect(() => {
        const fetchData = async () => {
            const config = {
                headers: {
                    'Accept': 'application/json',
                }
            };

            try {
                const res = await axios.get('http://localhost:8000/api/image/fetch-images', config);

                if (res.status === 200) {
                    setImages(res.data.images);
                }
            } catch(err) {
              console.log(err)
            }
        };

        fetchData();
    }, [updated]);

    const onFileChange = e => setImage(e.target.files[0]);

  return (
    <div className="flex justify-center items-center w-screen h-screen">
      <form className="flex justify-center items-center h-full" style={{ "flex": "column" }} encType="multipart/form-data">
        {/* <label htmlFor="file-upload" className="w-80 h-60 bg-gray-100 flex justify-center items-center rounded-xl bold"> Please drag files or click to select files.</label>
          <input id="file-upload" type="file" multiple style={{"opacity": 0}}/> */}
        <DragDrop setUpdated={setUpdated}/>
        <button className="w-40 h-10 bg-gray-100 flex justify-center items-center rounded-xl bold" style={{ "margin": "10px" }}>Get Result</button>
      </form>
      <div>
        <Image
          width={200}
          height={150}
            src={`http://localhost:3000${image.image}`}
          alt={"temp"}
        />
      </div>
    </div>
  );
}
