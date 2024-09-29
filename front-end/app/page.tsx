
import DragDrop from "./DragDrop";


export default function Home() {

  

  return (
    <div className="flex justify-center items-center w-screen h-screen">
      <form className="flex justify-center items-center h-full" style={{"flex":"column"}} encType="multipart/form-data">
          {/* <label htmlFor="file-upload" className="w-80 h-60 bg-gray-100 flex justify-center items-center rounded-xl bold"> Please drag files or click to select files.</label>
          <input id="file-upload" type="file" multiple style={{"opacity": 0}}/> */}
          <DragDrop/>
          <button className="w-40 h-10 bg-gray-100 flex justify-center items-center rounded-xl bold" style={{"margin":"10px"}}>Get Result</button>
      </form>
    </div>
  );
}
