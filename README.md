# Tutorial ML agents - Simulação de veículos autônomos em ambiente 3D usando reinforcement learning
## Introdução

No universo da criação de ambientes virtuais e simulações interativas, o desenvolvimento de agentes inteligentes tem se destacado como um campo de extrema relevância, trazendo  ampliação dos horizontes na Aprendizagem Profunda, inspirando assim futuras pesquisas. Para facilitar a criação e treinamento de agentes inteligentes, a Unity Technologies apresenta o ML-Agents, uma plataforma de alto nível que se destaca pela sua integração com Python e Unity. 
Uma das principais razões para a escolha do ML-Agents para o meu projeto é a sua integração entre a Unity e Python. A Unity é uma das plataformas de desenvolvimento de jogos mais amplamente utilizadas no mundo, e Python é uma linguagem de programação popular e acessível. Essa combinação permite que desenvolvedores sem conhecimento profundo em técnicas de Deep Learning ou Reinforcement Learning possam criar ambientes interativos e agentes inteligentes sem grandes obstáculos técnicos a partir de tutoriais e orientação assertiva. Contudo, por ser uma tecnologia nova, o seu uso não é trivial e não há muito material disponível na internet, principalmente em português, surgindo assim  a necessidade trazer caminhos que indiquem de que forma é possível iniciar e como animar cenários já existentes, democratizando a ferramenta.
A sua capacidade de fornecer resultados visuais impactantes é notória. Além de ser uma escolha pragmática para o desenvolvimento do projeto, o ML-Agents também se destaca pela sua capacidade de facilitar o ensino de Inteligência Artificial (IA) e Aprendizado por Reforço (RL). A sua integração com a Unity permite que alunos e entusiastas da área visualizem de forma concreta o funcionamento dos agentes e a evolução de suas habilidades. Isso torna o aprendizado mais envolvente e acessível, contribuindo para a disseminação do conhecimento em IA e RL.
Estabelecida as condições supracitadas, aqui serão abordadas explicações, dicas, possíveis erros e um tutorial prático para a construção de agentes inteligentes.  Com isso tornará o aprendizado mais fluido para próximas aplicações e possíveis desafios, agregando prática a objetos já estudados e de estudos futuros
### Reinforcement learning
Os sistemas de aprendizado por reforço funcionam como uma dupla, composta por um agente e um ambiente. O ambiente fornece informações que descrevem o momento presente do sistema, chamado de estado. O agente interage com o ambiente ao observar esse estado e decide sua ação com base nessas informações. O ambiente responde à ação tomada, transicionando para um novo estado e concedendo uma recompensa ao agente. Concluído o ciclo (estado → ação → recompensa), um passo no tempo é marcado. Esse ciclo continua até que o ambiente chegue a um estado final, como quando o problema é resolvido.
### Unity
A plataforma Unity é conhecida por sua capacidade de produzir jogos para diversos dispositivos, incluindo computadores, dispositivos móveis, realidade virtual e aumentada. Oferece ferramentas poderosas e um motor de jogo para criação, design e lançamento de experiências interativas, sendo amplamente utilizada por desenvolvedores profissionais e amadores na indústria de jogos e aplicativos. Para saber mais: https://unity.com/pt

### Possibilidades do ML-agents
O ML-Agents é um toolkit de código aberto do Unity para Machine Learning, possibilita a aplicação de simulações e ambientes virtuais no treinamento de agentes com habilidades inteligentes. O termo "ML-Agents" é uma abreviação de "Machine Learning Agents", que são essencialmente programas de software capazes de aprender a tomar decisões e realizar tarefas em um ambiente virtual. A plataforma ML-Agents, desenvolvida pela Unity Technologies, oferece uma estrutura poderosa para a criação e treinamento desses agentes. Ela se baseia em conceitos de aprendizado profundo (Deep Learning) e aprendizado por reforço (Reinforcement Learning), permitindo que os desenvolvedores construam ambientes virtuais complexos onde agentes podem aprender e melhorar suas habilidades ao longo do tempo. Entretanto  os agentes podem ser treinados também com Imitation Learning, Curriculum Learning, Deep Reinforcement Learning  e alguns outros métodos matemáticos presentes em bibliotecas. Em resumo, a comunicação e o ambiente possuem as seguintes partes.

Em resumo, a comunicação e o ambiente possuem as seguintes partes:
Ambiente de Aprendizagem: É a cena do Unity onde os agentes interagem e aprendem. A configuração desse ambiente depende dos objetivos, pode-se reutilizar a mesma cena para treinar e testar agentes ou criar um cenário específico para treinamento em jogos complexos.

API Python de Baixo Nível: Uma interface independente do Unity, parte do pacote Python mlagents_envs, para interagir com o ambiente de aprendizagem. Usada no treinamento  em Python para comunicar e controlar a Academia, mas também pode ser utilizada para outros propósitos, como empregar o Unity em algoritmos de aprendizado de máquina personalizados.

Comunicador Externo: Conecta o Ambiente de Aprendizagem à API Python de Baixo Nível, integrado ao próprio Ambiente de Aprendizagem.

Python Trainers: Contém algoritmos de aprendizado de máquina implementados em Python, permitindo o treinamento dos agentes. Interface exclusiva com a API Python de Baixo Nível.


Agentes: são ligados a GameObjects no Unity, assumindo a responsabilidade de observar, agir e recompensar conforme apropriado. Cada agente está associado a um Comportamento específico.

Comportamento: Um Comportamento é um conjunto de atributos do agente, como o número de ações possíveis, identificado por um único nome. Funciona como uma função que recebe observações e recompensas do Agente, gerando ações. Pode ser de três tipos: Aprendizagem (em treinamento), Heurística (baseado em regras codificadas) ou Inferência (com rede neural treinada). Um Comportamento de Aprendizagem é preparado para treino, enquanto o Heurístico opera por regras e o de Inferência usa redes neurais treinadas. Após o treino, um Comportamento de Aprendizagem se torna um de Inferência.

## Tutorial

Primeiro, é importante saber que existe toda uma documentação específica para o ML agents, e pode ser acessado pelo link COLOCAR LINK. Porém ela é muito técnica, e leva em consideração que quem tem contato com ela já possui conhecimento de Python, Unity, computação e git. Logo, é indicado que siga o tutorial aqui citado, como forma de poder visualizar os resultados e o que está sendo lido e entendido, e também servir de inspiração após ver o primeiro projeto pronto. No entanto, após isso, é necessária a leitura da documentação, bem como aplicação dos tutoriais disponibilizados pela própria unity.

#### Instalando Python e Unity
A primeira ação a se tomar é baixar o Unity, o tutorial de como baixar o Unity do canal Crie seus jogos pode ser utilizado, através do link - https://www.youtube.com/watch?v=wHzu5Cf9ig4&ab_channel=CrieSeusJogos.

Para baixar o Python, o turial do canal Código logo pode ser usado, porém nesse tutorial é requerido o python 3.7.9 - https://www.youtube.com/watch?v=0pG4NrucQR4&ab_channel=C%C3%B3digoLogo.

#### Baixando projeto inicial no github
Basta rolar para cima e seguir o caminho da figuraabaixo, após isso extraia os arquivos.  O ambiente foi produzido por Sebastian Schuchmann e seu uso foi autorizado, bem como o tutorial em português foi autorizado.
<div align="center">
<img src="https://github.com/joaoOldenburg/PI_3_RL_IFSC/assets/111868475/242f7755-4398-42ac-ac07-5c79f6c442b8" width="300px" />
</div>

#### Abrindo o projeto no unity
Para abrir o projeto no Unity é muito simples, basta abrir o unity e clicar em “add”, após isso em “add project from disk” e navegar até onde foi baixado o arquivo do git hub e selecionar.

<div align="center">
<img src="https://github.com/joaoOldenburg/PI_3_RL_IFSC/assets/111868475/965fbf7d-ae7c-4fd3-9292-9eee0d18d0ed" width="600px" />
</div>
Quando o projeto for aberto o unity criará uma caixa de diálogo dizendo que a versão atual instalada não é a versão em que o projeto foi criado, visto isso será necessário selecionar e instalar outra versão do unity, nesse caso selecione a versão 2019.3.5f1 conforme imagem acima.

### Instalando o ambiente e importando as bibliotecas Python

Após os passos anteriores estarem feitos será necessário criar o ambiente python em que o treinamento ocorrerá, esse ambiente Python é um espaço virtual que permite que seja instalado e usado pacotes Python sem interferir com outros projetos Python em sistema, logo tudo que será instalado funcionará exclusivamente no ambiente criado.
Para isso vá até o projeto baixado e extraído, digite cmd na barra de pesquisa e aperte  a tecla enter, conforme abaixo, após isso o prompt de comando do windows deve abrir já no diretório onde o ambiente Python deve ser criado, na pasta do projeto.
<div align="center">
<img src="https://github.com/joaoOldenburg/PI_3_RL_IFSC/assets/111868475/5810f6a1-d9bd-4ead-96be-99da61d88981" width="600px" />
</div>
Após isso, o primeiro comando a ser digitado deve ser 'py -m venv venv', esse comando criará dentro do arquivo baixado do github uma pasta chamada venv, esse é o ambiente virtual. Mas nada deve ser feito com a pasta no momento, apenas continue com os próximos comandos no mesmo prompt.

<div align="center">
<img src="https://github.com/joaoOldenburg/PI_3_RL_IFSC/assets/111868475/29ac2655-d193-4c4c-857d-5e77ca2a327c" width="600px" />
</div>

Agora é necessário ativar o ambiente virtual para possibilitar que tudo seja feito nele, com o comando 'venv\scripts\activate'
<div align="center">
<img src="https://github.com/joaoOldenburg/PI_3_RL_IFSC/assets/111868475/0144da8d-e031-43ce-bd78-e7957dc13691" width="600px" />
</div>

Dois  comandos devem ser executados nessa ordem: 'pip install protobuf==3.20.3', 'pip3 install mlagents==0.16.0'. É possível notar que eles terão tempos diferentes no seu download, ambos são bibliotecas, também que eles possuem versões especificadas, isso é indicado pelos dois símbolos de igual em sequência, os quais informam que a biblioteca a ser baixada deve ser aquela  do número especificado em sequência. Alguns avisos podem ocorrer, porém podem ser ignorados se todos os passos to tutorial forem seguidos.

### Alterando o projeto Unity

A partir de agora será necessário fazer modificações no arquivo dentro do Unity, e nos seus complementares, os scripts, esses serão modificados no Visual Studio utilizando a linguagem c#.
Após abrir o projeto no unity, será necessário entrar na pasta scenes e clicar duas vezes no arquivo main, conforme abaixo, isso fará o cenário do treinamento aparecer. Também é preciso seguir o caminho para verificar o pacote ML agents no unity: Window, Package manager, pesquise por ML agents e instale. Caso a versão 1.0.0 já esteja instalada, não há necessidade de modificação, é possível apenas seguir com o tutorial.

<div align="center">
<img src="https://github.com/joaoOldenburg/PI_3_RL_IFSC/assets/111868475/e5b4c73f-521e-477c-9402-15bbfe4e9c46" width="600px" />
</div>

Após isso verifique na figura abaixo, o caminho onde todas as modificações serão realizadas, a setas verdes indicam o caminho do mouse para chegar na aba inspector do agente. A janela do Inspetor é usada para visualizar e editar propriedades e configurações de quase tudo no Unity Editor, incluindo objetos de jogo, componentes do Unity, ativos, materiais e configurações e preferências no editor.
É possível notar que no inspector existe script chamado jumper, esse é um código C# responsável por fazer o carro pular, no momento pode ser usado apenas iniciando o jogo no botão play do Unity, e sendo utilizado pelo jogador usando a tecla ‘espaço’. Esse script precisará ser modificado para possibilitar a autonomia do agente
<div align="center">
<img src="https://github.com/joaoOldenburg/PI_3_RL_IFSC/assets/111868475/1203acf8-5627-4886-a493-36a9cb27bd84" width="600px" />
</div>

Para iniciar as modificações o script jumper ele deverá ser aberto no visual studio, seguindo o caminho:  Assets, Scripts e Jumper, conforme figura abaixo.

<div align="center">
<img src="https://github.com/joaoOldenburg/PI_3_RL_IFSC/assets/111868475/d501fa8d-ca03-4c3f-99a6-33e3d80d5d48" width="400px" />
</div>

Para transformar o veículo presente na cena em um agente, é preciso modificar de monobehaviour, comportamento único, para Agent, presente abaixo. Também será necessário através do botão, add compenente, da figura acima, adicionar o componente Behavior parameters, e alterar seu nome para algo significativo. O componente Behavior Parametes permite definir as propriedades de comportamento do agente, ou seja, é aqui que é definida grande parte do processo de aprendizado, podendo trazer resultados positivos ou negativos a partir de escolhas.

<div align="center">
<img src="https://github.com/joaoOldenburg/PI_3_RL_IFSC/assets/111868475/60669a01-bec1-4bb4-a10e-51e95a382b70" width="400px" />
</div>

Outras modificações necessárias no componente Behavior  Parameter  serão mostradas neste parágrafo. Primeiro devemos modificar o Branch 0 size para dois, e manter o branch size em  um, isso ocorre porque o agente apenas pode tomar duas ações, pular e não fazer nada, mas não as duas ao mesmo tempo. Então, o campo Space Size deve ser passado para zero, pois não será necessário adicionarmos códigos de observação, em próximos passos será adicionado outro componente que fará o trabalho de observar, ser os olhos do agente. Os demais campos devem ser mantidos, resultando nas modificações da figura abaixo.

<div align="center">
<img src="https://github.com/joaoOldenburg/PI_3_RL_IFSC/assets/111868475/14460baf-aad4-4938-8a60-f9bd43adddf6" width="400px" />
</div>


Agora a visão do agente deve ser adicionada, o sensor,  para isso é adicionado o componente Ray Perception Sensor 3D, através do botão add componente. Esse componente mede a distância entre o agente e o outro objeto objeto de cena desejado, nesse caso é necessário detectar  uma única tag pertencente a um objeto de cena o Mover, então o campo Size deve ser alterado para um e o campo Element 0 deve ser alterado para Mover. Após o campo Ray Per Direction deve ser alterado para zero, pois conforme pode ser visto na cena, somente será necessário observar os carros que virão pela frente. Ray Length para 50, permitindo o agente perceber até mais longe, e Start Vertical Offset e End Vertical Offset para 0.5, isso levanta o raio de percepção, impedindo que esbarre no chão.
O componente ficará conforme a figura abaixo.

<div align="center">
<img src="https://github.com/joaoOldenburg/PI_3_RL_IFSC/assets/111868475/9fae60b2-7e5c-4ebe-a85a-8f3098c3717b" width="400px" />
</div>

Agora, para conectar tudo, a alteração será feita no script Jumper. Será definido, conforme agentes baseados em reinforcement learning, as ações possíveis, onde o episódio inicia e termina, quais as recompensas associadas. Primeiramente, três métodos serão adicionados, OnActionReceived que receberão as ações, OnEpisodeBegin que informará o início do episódeo, e Initialize que informa onde os objetos de cena iniciam, por exemplo.
Confome a figura abaixo. O que estava dentro do método Awake deve ser passado para o Initialize, e dentro do OnActionReceived a ação a ser executada.

```csharp
public override void OnActionReceived(float[] vectorAction)  
{  
 if (Mathf.FloorToInt(vectorAction[0]) == 1)  
  Jump();  
}
```
```csharp
public override void OnEpisodeBegin()
{
    Reset();
}
```
```csharp
public override void Initialize()  
{  
 rBody = GetComponent<Rigidbody>();  
 startingPosition = transform.position;  
}
```
Também é necessário alterar o input para o método heuristic, em vez do update, devido a uma necessidade de gerenciamento interno do método OnactionReceived. Contudo, é preciso solicitar uma decisão, e isso será feito pel adição do método Fixedupdate, que para otimização só solicitará decisões se o pulo puder ser realizado.
```csharp
public override void Heuristic(float[] actionsOut)
{
    actionsOut[0] = 0;

    if (Input.GetKey(jumpKey))
        actionsOut[0] = 1;
}

```
```csharp
private void FixedUpdate()
{
    if (jumpIsReady)
        RequestDecision();
}
```

Agora, se faz necessário adicionar recompensas para o aprendizado do agente, a recompensa negativa deve ser adicionada caso o agente colida em outros carros, e o episódio também deve terminar com isso, dessa forma o método Oncollissionenter deve ser modificado. Para as recompensas positivas, indicando o comportamento a ser seguido, devem ser adicionadas a um gatilho, este ocorre sempre que um carro passa pelo agente sem colisão, com isso o método Ontriggerenter deve ser modificado. O Script final pode ser baixado, também o projeto finalizadoh(ttps://github.com/Sebastian-Schuchmann/A.I.-Jumping-Cars-ML-Agents-Example/tree/TutorialCompleted), é importante atentar ao script finalizado para encontrar possíveis erros

```csharp
private void OnCollisionEnter(Collision collidedObj)
{
    if (collidedObj.gameObject.CompareTag("Street"))
        jumpIsReady = true;

    else if (collidedObj.gameObject.CompareTag("Mover") || collidedObj.gameObject.CompareTag("DoubleMover"))
    {
        AddReward(-1.0f);
        EndEpisode();
    }
}

private void OnTriggerEnter(Collider collidedObj)
{
    if (collidedObj.gameObject.CompareTag("score"))
    {
        AddReward(0.1f);
        score++;
        ScoreCollector.Instance.AddScore(score);
    }
}
}
```
### Rodando o treinamento

Os hiperparâmetros de um  agente que utiliza Deep reinforcement learning são vários, as redes do ML agents possuem modos default, ou seja, caso não cadastre nada assumirão os valores padrões. A modificação desses parâmetros podem ser vistos na figura abaixo, e podem ser modificado através de um arquivo txt, neste projeto é encontrado na pasta trainer_config, no arquivo trainer_config.txt.

<div align="center">
<img src="https://github.com/joaoOldenburg/PI_3_RL_IFSC/assets/111868475/ba49c3d1-3a9e-4af6-9bd4-37c86ec0f6a3" width="400px" />
</div>

Para iniciar o treinamento se deve voltar a aba do CMD que foram instaladas os pacotes Python, caso tenha fechado, basta ir no arquivo, digitar CMD, conforme já explicado, desta vez bastará ativar o ambiente através do comando venv\scripts\activate. Dentro do ambiente python devemos rodar o comando  'mlagents-learn TrainerConfig/trainer_config.yaml --run-id=Nome_do_treinamento', é possível notar que é necessário passar o caminho do arquivo de hiperparâmetros e também passar um nome para o treinamento, pois será criado uma pasta com esse nome, para caso posteriormente seja necessário revisitar os resultados do treinamento. Após rodar o comando, a logo da Unity deve aparecer, solicitando que seja apertado o play dentro do Unity, caso o play não seja apertado no Unity, o treinamento não iniciará. Pronto a rede está treinando.

<div align="center">
<img src="https://github.com/joaoOldenburg/PI_3_RL_IFSC/assets/111868475/d68ea85c-647a-4eec-bc04-132cb04e92e1" width="700px" />
</div>

Para acelerar o tempo de treinamento é possível copiar o envoriment, fazendo múltiplos agentes, funcionará da mesma forma, conforme  figura abaixo.

<div align="center">
<img src="https://github.com/joaoOldenburg/PI_3_RL_IFSC/assets/111868475/6c16ce00-353a-42c6-adff-1019f09d2f4b" width="700px" />
</div>

O pacote tensorflow que é instalado automaticamente pelo ML agents disponibiliza uma dashboard completa para verificação do treinamento, que pode ser acessada abrindo o CMD, conforme explicado em tópicos acima, rode comando 'tensorboard --logdir=summaries'. É importante que seja aberto após alguns minutos de treinamento, ou quando o high score do agente já estiver alto, quando rodar o comando aparecerá um link, esse deverá ser colado em qualquer navegador, assim será acessada a dahsboard. Conforme a figura abaixo. Após o treinamento rodar por pelo menos 10 minutos, ou um high score de pelo menos 300, o treinamento pode ser parado clicando no botão de pause do Unity, assim será garantido uma rede reasoávlel para o próximo passo.

<div align="center">
<img src="https://github.com/joaoOldenburg/PI_3_RL_IFSC/assets/111868475/95e51609-ffb6-4aae-a6e2-6ecd7a55b9a1" width="700px" />
</div>

### Adicionando uma rede já treinada ao agente

Após o treinamento uma pasta chamada models é criada, dentro da pasta terão os treinamentos rodados separados por outras pastas com o nome que foi definido no passo anterior, basta arrastar a pasta para dentro do Unity na pasta Assets, figura abaixo. 

<div align="center">
<img src="https://github.com/joaoOldenburg/PI_3_RL_IFSC/assets/111868475/c7c8db1d-0925-402f-b65a-e33b91308262)" width="700px" />
</div>

Abrindo a pasta arrastada ao Unity a rede treinada estará presente, ela poderá ser adicionada dentro do componente Behavior Parameters, no campo Model, figura abaixo, o campo Behavior Type deverá ser alterado para Inference. Apertando play apenas no Unity é possível assistir a rede treinada em funcionamento.

<div align="center">
<img src="https://github.com/joaoOldenburg/PI_3_RL_IFSC/assets/111868475/72adc9f5-021c-4f43-87eb-8731c505cfb5" width="700px" />
</div>

### Possível erro encontrado
Durante o processo de aprendizado, alguns outros erros foram encontrados, porém todos estavam ligados a compatibilidade entre versões de pacotes usados, exceto o erro citado na sequência, devido a isso a solução de encontra abaixo.
O erro not have Longh Path Support Enabled que pode ocorrer na instalação dos pacotes Python. Para solucionar basta seguir os seguintes comandos, abra o Editor de Registro do Windows, é possível fazer isso pressionando Win + R, digitando “regedit” e pressionando Enter. Navegue até a seguinte chave do registro: HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem Procure uma entrada chamada “LongPathsEnabled”. Se ela não existir, você pode criá-la. Clique com o botão direito no painel direito, selecione “Novo” e depois escolha “Valor DWORD (32 bits)”. Nomeie o novo valor DWORD como “LongPathsEnabled”. Clique duas vezes em “LongPathsEnabled” e defina seu valor como 1. Feche o Editor de Registro.
<div align="center">
<img src="https://github.com/joaoOldenburg/PI_3_RL_IFSC/assets/111868475/d2f89bba-eb68-4c11-a113-cf43f66f63d2" width="600px" />
</div>



