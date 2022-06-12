import axios from "axios";
import React from "react";
import "./FileUpload.css";
class FileUpload extends React.Component {
  constructor() {
    super();
    this.state = {
      selectedFile: "",
      imagePreviewUrl: "",
      name: "",
      podstatus:""
    };
    this.handleInputChange = this.handleInputChange.bind(this);
  }

  
  handleInputChange(event) {
    event.preventDefault();
    let reader = new FileReader();
    let file = event.target.files[0];
    reader.onloadend = () => {
      this.setState({
        selectedFile: file,
        //name: event.target.data,
        imagePreviewUrl: reader.result
      });
    };
    reader.readAsDataURL(file);
  }

  submit() {
    const data = new FormData();
    
    data.append("file", this.state.selectedFile);
    console.warn(this.state.selectedFile);
    let url = "http://127.0.0.1:8000/api/predict";

    axios
      .post(url, data)
      .then(res => {
        // then print response status
        this.setState({ name: res.data });
        console.log(this.name)
      })
      .catch(error => {
        this.setState({
            podstatus: 'Stop'
        });
        console.log("error")
    })
  }

  render() {
    return (
      <div className = "main">
        <div className="form-row">
          <div className="form-group-col-md-6">
                  <label className="text-white">Select File :</label>
                  <input type="file" className="form-control" name="upload_file"onChange={this.handleInputChange}/>
          </div>
          <img src={this.state.imagePreviewUrl} alt="description"height={ 500 }width={ 500 }/>
        </div>
        <div className="col-md-6">
        <button type="submit" className="btn btn-dark"onClick={() => this.submit()}>
          名前確認
        </button>
      </div>
      <div className = "result">
        {this.state.name}
      </div>
    </div>
    );
  }
}

export default FileUpload;
