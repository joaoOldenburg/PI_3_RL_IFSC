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

#### Possibilidades do ML-agents
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

### Instalando Python e Unity
A primeira ação a se tomar é baixar o Unity, o tutorial de como baixar o Unity do canal Crie seus jogos pode ser utilizado, através do link - https://www.youtube.com/watch?v=wHzu5Cf9ig4&ab_channel=CrieSeusJogos.

Para baixar o Python, o turial do canal Código logo pode ser usado, porém nesse tutorial é requerido o python 3.7.9 - https://www.youtube.com/watch?v=0pG4NrucQR4&ab_channel=C%C3%B3digoLogo.

### Baixando projeto inicial no github
Basta rolar para cima e seguir o caminho da figuraabaixo, após isso extraia os arquivos.  O ambiente foi produzido por Sebastian Schuchmann e seu uso foi autorizado, bem como o tutorial em português foi autorizado.
<div align="center">
<img src="https://github.com/joaoOldenburg/PI_3_RL_IFSC/assets/111868475/242f7755-4398-42ac-ac07-5c79f6c442b8" width="400px" />
</div>


