function refresh(){
    fetch("/det")
        .then(res => res.json())
        .then(function(jdata){
            console.log(jdata);
            document.getElementById("time").innerText = jdata["time"];
            if(jdata["message"] == "OK"){
                fetch("/img/" + jdata["filename"])
                    .then(res => res.blob())
                    .then(function(blob){
                        if(blob){
                            document.getElementById("img").setAttribute("src", URL.createObjectURL(blob));
                        }
                        
                    });
                    // .then(function(res){
                    //     if(res.status == 200){
                    //         var blob = res.blob();
                    //         document.getElementById("img").setAttribute("src", URL.createObjectURL(blob));
                    //     }
                    // })
            }
        })
}

refresh();
setInterval("refresh()", 1000/25);