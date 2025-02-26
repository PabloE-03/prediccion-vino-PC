const animacion = ()=>{
    const select = document.getElementById("select")
    const caja_1 = document.getElementsByClassName("animacion-1")[0]
    const regresion = document.getElementsByClassName("regresion")[0]
    const caja_2 = document.getElementsByClassName("animacion-2")[0]
    const knn = document.getElementsByClassName("knn")
    const circulo = document.getElementsByClassName("circulo")[0]
    if(select.value=="regresion")
    {
        caja_2.style.transitionDelay = "0s"
        circulo.style.transitionDelay = "0s"
        knn.item(0).style.transitionDelay = "0s";
        knn.item(1).style.transitionDelay = "0s";
        knn.item(2).style.transitionDelay = "0s";
        caja_2.style.width = "0vw";
        caja_2.style.height = "0vh";
        caja_2.style.borderColor = "transparent";
        for(let item = 0;item<knn.length;item++)
        {
            knn.item(item).style.width = "0%";
            knn.item(item).style.height = "0vh";
        }
        circulo.style.borderColor = "transparent";

        caja_1.style.transitionDelay = "1s"
        regresion.style.transitionDelay = "1.8s";
        caja_1.style.width = "2.5vw";
        caja_1.style.height = "5vh";
        caja_1.style.borderColor = "black";
        regresion.style.width = "100%";
        regresion.style.borderColor = "darkred"; 
    }
    else
    {
        caja_2.style.transitionDelay = "1s"
        circulo.style.transitionDelay = "3s"
        knn.item(0).style.transitionDelay = "2s";
        knn.item(1).style.transitionDelay = "2.2s";
        knn.item(2).style.transitionDelay = "2.4s";

        caja_1.style.transitionDelay = "0s"
        regresion.style.transitionDelay = "0s";
        caja_1.style.width = "0vw";
        caja_1.style.height = "0vh";
        caja_1.style.borderColor = "transparent";
        regresion.style.width = "0%";
        regresion.style.borderColor = "transparent";

        caja_2.style.width = "3.5vw";
        caja_2.style.height = "6vh";
        caja_2.style.borderColor = "black";


        for(let item = 0;item<knn.length;item++)
        {
            knn.item(item).style.width = "20%";
            knn.item(item).style.height = "1.5vh";
        } 
        // circulo.style.width = "40%";
        // circulo.style.height = "4.5vh";
        circulo.style.borderColor = "black";
    }
}

const evaluate_wine = async()=>{
    const select = document.getElementById("select");
    const input = document.getElementsByTagName("input");
    let valores = {
        "fixed acidity":input.item(0).value,
        "volatile acidity":input.item(1).value,
        "citric acid":input.item(2).value,
        "chlorides":input.item(3).value,
        "total sulfur dioxide":input.item(4).value,
        "density":input.item(5).value,
        "sulphates":input.item(6).value,
        "alcohol":input.item(7).value
    };
    let url = select.value=="regresion" ? "http://localhost:5002/regressor" : "http://localhost:5002/classifier";

    const response = await fetch(url,{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify(valores)
    })

    if(response.ok)
    {
        const data = await response.json()
        const calidad = document.getElementById("calidad")
        calidad.textContent = "El vino introducido es de calidad "+data.prediction
        const content = document.getElementsByClassName("content")[0]
        if(data.prediction==3)
        {
            content.style.width = "20%";
            content.style.backgroundColor = "darkred";
        }
        else if(data.prediction==4)
        {
            content.style.width = "35%";
            content.style.backgroundColor = "darkorange";
        }
        else if(data.prediction==4)
        {
                content.style.width = "45%";
                content.style.backgroundColor = "orange";
        }
        else if(data.prediction==5)
        {
            content.style.width = "55%";
            content.style.backgroundColor = "darkgoldenrod";
        }
        else if(data.prediction==6)
        {
            content.style.width = "60%";
            content.style.backgroundColor = "yellowgreen";
        }
        else if(data.prediction==7)
        {
            content.style.width = "85%";
            content.style.backgroundColor = "green";
        }
        else if(data.prediction==8)
        {
            content.style.width = "100%";
            content.style.backgroundColor = "forestgreen";
        }
    }
    else
    {
        const calidad = document.getElementById("calidad")
        calidad.textContent = "No se ha podido procesar la solicitud"
        const content = document.getElementsByClassName("content")[0]
        content.style.width = "100%";
        content.style.backgroundColor = "darkred";
    }   
}