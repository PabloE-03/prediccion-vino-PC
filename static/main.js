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

const evaluate = async()=>{
    const select = document.getElementById(select);
    const input = document.getElementsByTagName("input");
    let valores = {
        "fixed acidity":input.item(0),
        "volatile acidity":input.item(1),
        "citric acid":input.item(2),
        "chlorides":input.item(3),
        "total sulfur dioxide":input.item(4),
        "density":input.item(5),
        "sulphates":input.item(6),
        "alcohol":input.item(7)
    };
    let url = select.value=="regresion" ? "/regression" : "/classifier";

    const response = await fetch(url,{
        method:"POST",
        body:JSON.stringify(valores)
    })

    if(response.ok)
    {
        const data = await response.json()
        return undefined
    }
    else
    {
        return false
    }
}