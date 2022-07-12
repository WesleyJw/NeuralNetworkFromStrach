#include <stdio.h>
#include <stdlib.h>
#include <math.h>


/*
                        >>>  Estatistica Computacional 2016.1  <<<
                        >>>             Borko Stosic           <<<
                        >>>       Jose Wesley Lima Silva       <<<
    Reconhecimento de Padroes e Previsao com Redes Neurais Artificiais - Perceptron e Adalaine
*/

    //Declaracao do prototipo das funcoes

    float PerceptronTrain(float **xj, float *dj, float *wj, int n_entradas);
    float PerceptronPredict(float *pesos_ajustados, int k_entradas);
    float AdalaineTrain(float **xj, float *dj, float *wj, int n_entradas);
    float AdalainePredict(float *pesos_ajustados, int k_entradas, char tipo_funcao);
    //-------------------------------------------------------------------------

int main()
{


    //Declaracao de variaveis

    float **vlr_exemplo = (float**) malloc(sizeof(float*)*1);
    float *vlr_desejado = (float*) malloc(sizeof(float)*1);
    float *pesos_sinapticos = (float*) malloc(sizeof(float)*1);
    float valor,peso_wj;
    int n,k, n_teste;
    int contadorValor=0;
    int contadorLinhas = 0;

    //Abrindo arquivo

    printf("\n\tInforme o numero de entradas em cada exemplo:  ");
    scanf("%d",&k);

    FILE *arquivo;

    printf("\n\t    Abrindo arquivo...\n");

    /*
    Bancos de dados

    1) SistemaAlerta.txt
    2) NivelRio.txt
    */

    arquivo = fopen("NivelRio.txt", "r");

    if(arquivo == NULL){
        printf("\tErro ao abrir arquivo!!!\n");
        exit(0);
    }

    //Lendo Arquivo

    while (!feof(arquivo)){

        if (contadorValor == 0){
            fscanf (arquivo, "%f", &valor);
            vlr_exemplo[contadorLinhas] = (float*) malloc(sizeof(float)*k);
            vlr_exemplo[contadorLinhas][contadorValor] = valor;
            contadorValor++;
            }
        if(contadorValor > 0 && contadorValor < k){
            fscanf (arquivo, "%f", &valor);
            vlr_exemplo[contadorLinhas][contadorValor] = valor;
            contadorValor++;
        }
        if (contadorValor == k){
            fscanf (arquivo, "%f", &valor);
            vlr_desejado[contadorLinhas] = valor;
            contadorValor = 0;
            contadorLinhas++;

            vlr_desejado = (float*) realloc(vlr_desejado,(contadorLinhas+1)*sizeof(float));
            vlr_exemplo = (float**)realloc(vlr_exemplo,(contadorLinhas+1)*sizeof(float*));
            }
    }
    for (n=0; n < k; n++){
        printf("\n\tInforme o peso peso_sinaptico[%d]:  ",n);
        scanf("%f", &peso_wj);
        pesos_sinapticos[n] = peso_wj;
        pesos_sinapticos = (float*) realloc(pesos_sinapticos,(n+1)*sizeof(float));
    }

    fclose(arquivo);

    //Reconhecimento de Padroes com Perceptron

    /*
    Chamada da funcao PerceptronTrain
    */
    //PerceptronTrain(vlr_exemplo,vlr_desejado,pesos_sinapticos,k);


    /*
    Chamada da funcao AdalaineTrain
    */
    AdalaineTrain(vlr_exemplo,vlr_desejado,pesos_sinapticos,k);

    return 0;
}

float PerceptronTrain(float **xj, float *dj, float *wj, int n_entradas){

    float alpha, net, yj, erro, erro_calculado, erro_calculado2, EQM, predicao;
    int n_ciclos, ciclos, i, j, n_exemplo;
    char escolha;
    EQM = 10;

    printf("\n\nInforme o numero de exemplos que seram utilizados para o treinamento da Rede: ");
    scanf("%d", &n_exemplo);
    printf("\n\nInforme o valor de alpha (taxa de aprendizagem): ");
    scanf("%f", &alpha);
    printf("\n\nInforme o numero de ciclos desejados: ");
    scanf("%d", &ciclos);
    printf("\n\nInforme o erro minimo desejado: ");
    scanf("%f", &erro);

    printf("Iniciando processo iterativo...\n");

    n_ciclos = 1;
    while(n_ciclos <= ciclos && EQM > erro){

        erro_calculado2 = 0;
        for(i=0; i < n_exemplo; i++){
            float net = 0.0;

            for(j=0; j < n_entradas; j++){
                net = net + (wj[j]*xj[i][j]);
            }
            if(net >= 0){
                yj = 1.;
            }else{
                yj = 0.;
            }
            erro_calculado = yj - dj[i];
            erro_calculado2 = erro_calculado2 + pow(erro_calculado,2);
            if(yj != dj[i]){

                for(j=0; j < n_entradas; j++){
                    wj[j] = wj[j] + alpha*(dj[i]-yj)*(xj[i][j]);
                }
                system("clear");
            }else{
                continue;
            }
        }
        EQM = erro_calculado2/i;
        n_ciclos++;
    }

    printf("\n\tA Rede esta treinada!\n\n\t Deseja predizer valores para uma nova entrada?\n\n");
    printf("\tDigite (s) ou (n):");
    scanf("  %c", &escolha);

    if(escolha == 's'){
        predicao = PerceptronPredict(wj, n_entradas);
        printf("\n\n\tO valor estimado e: >>> %.4f \n\n", predicao);
        printf("\n\t\tPrograma Finalizado!!\n\n");
    }else{
        printf("\n\t\tPrograma Finalizado!!\n\n");
    }

    return 0;
}

float PerceptronPredict(float *pesos_ajustados, int k_entradas){

    float *nova_entrada = (float*) malloc(sizeof(float)*1);
    float prediction, net;
    int k, n;

    n = 0;
    while(n < k_entradas){
        printf("\n\n\tInforme o valor da entrada %d (0=bias, informe 1):  ", n);
        scanf("%f", &nova_entrada[n]);
        nova_entrada = (float*) realloc(nova_entrada,(n+1)*sizeof(float));
        n++;
    }

    system("clear");

    net = 0;
    for (k=0; k < k_entradas; k++){
        net = net + pesos_ajustados[k]*nova_entrada[k];
    }

    if(net >= 0){
        prediction = 1;
    }else{
        prediction = 0;
    }
    return prediction;
}

float AdalaineTrain(float **xj, float *dj, float *wj, int n_entradas){

    float alpha, net, yj, erro, erro_calculado, erro_calculado2, EQM, predicao;
    int n_ciclos, ciclos, i, j, n_exemplo,n_teste;
    char escolha, funcao;
    EQM = 10;

    printf("\n\nInforme o numero de exemplos que seram utilizados para o treinamento da Rede: ");
    scanf("%d", &n_exemplo);
    printf("\n\nInforme o valor de alpha (taxa de aprendizagem): ");
    scanf("%f", &alpha);
    printf("\n\nInforme o numero de ciclos desejados: ");
    scanf("%d", &ciclos);
    printf("\n\nInforme o erro minimo desejado: ");
    scanf("%f", &erro);

    printf("\nEscolha a funcao de ativacao para realizar o treinamento da Rede!\n\n");
    printf("\tDigite (l) funcao 'linear' ou (s) funcao 'sigmoide logistica':");
    scanf("  %c", &funcao);




    printf("Iniciando processo iterativo...\n");

    n_ciclos = 1;
    while(n_ciclos <= ciclos && EQM > erro){

        erro_calculado2 = 0;
        for(i=0; i < n_exemplo; i++){
            float net = 0.0;

            for(j=0; j < n_entradas; j++){
                net = net + (wj[j]*xj[i][j]);
            }
            if( funcao == 'l'){
                yj = net;
            }
            if ( funcao == 's'){
                yj = 1.0/(1.0+exp(-net));
            }
            erro_calculado = yj - dj[i];
            erro_calculado2 = erro_calculado2 + pow(erro_calculado,2);
            if( funcao == 'l'){
                if(yj != dj[i]){
                    for(j=0; j < n_entradas; j++){
                        wj[j] = wj[j] + alpha*(dj[i]-yj)*(xj[i][j]);
                    }
                }else{
                    continue;
                }
            }
            if( funcao == 's'){
                if(yj != dj[i]){
                    for(j=0; j < n_entradas; j++){
                        wj[j] = wj[j] + alpha*(dj[i]-yj)*(xj[i][j])*yj*(1.0-yj);
                    }
                }
            }else{
                continue;
            }
            //system("clear");
        }
        EQM = erro_calculado2/i;
        printf("\n\t O Errro quadrado medio no ciclo [%d] e >>> %f\n", n_ciclos, EQM);
        n_ciclos++;
    }

    printf("\n\tA Rede esta treinada!\n\n\t Deseja predizer valores para uma nova entrada?\n\n");
    printf("\tDigite (s) ou (n):");
    scanf("  %c", &escolha);

    if(escolha == 's'){
        predicao = AdalainePredict(wj, n_entradas, funcao);
        printf("\n\n\tO valor estimado e: >>> %.4f \n\n", predicao);
        printf("\n\t\tPrograma Finalizado!!\n\n");
    }else{
        printf("\n\t\tPrograma Finalizado!!\n\n");
    }

    return 0;
}

float AdalainePredict(float *pesos_ajustados, int k_entradas, char tipo_funcao){

    float *nova_entrada = (float*) malloc(sizeof(float)*1);
    float prediction, net;
    int k, n;

    n = 0;
    while(n < k_entradas){
        printf("\n\n\tInforme o valor da entrada %d (0=bias, informe 1):  ", n);
        scanf("%f", &nova_entrada[n]);
        nova_entrada = (float*) realloc(nova_entrada,(n+1)*sizeof(float));
        n++;
    }

    system("clear");

    net = 0;
    for (k=0; k < k_entradas; k++){
        net = net + pesos_ajustados[k]*nova_entrada[k];
    }

    if(tipo_funcao == 'l'){
        prediction = net;
    }
    if(tipo_funcao == 's'){
        prediction = 1.0/(1.0+exp(-net));
    }

    return prediction;
}
