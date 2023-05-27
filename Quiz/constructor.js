
function Question(questionText, answerOptions, correctAnswer){
    this.questionText = questionText;
    this.answerOptions = answerOptions;
    this.correctAnswer = correctAnswer;
}

Question.prototype.checkedAnswer = function(answer){
    return answer === this.correctAnswer
    
}

let Questions =[
    new Question(
                "What language is this quiz program written in?",
                {"a":"Python","b":"CSS","c":"JavaScript","d":"C++"},
                "c"),

    new Question("How many days in 1 year?",
    {"a":"365 day","b":"340 day","c":"600 day",},
    "a"),
    new Question("What's up ?",
    {"a":"I'm Fine","b":"I'm Bad","c":"I'm so bad."},
    "a"),
    new Question("lorem lorem lorem ?",
    {"a":"node","b":"python","c":"java"},
    "c")
];


function Quiz(Questions){
    this.Questions = Questions;
    this.questionIndex = 0;
}

Quiz.prototype.getQuestion = function(){
    return this.Questions[this.questionIndex];
}
optionList = document.querySelector(".option-list");
const correctIcon = '<div class="icon"><i class="fas fa-check"></i></div>';
const incorrectIcon = '<div class="icon"><i class="fas fa-times"></i></div>';

function showQuestion(question){
    let questionTxt = `<span>${question.questionText}<span/>`;
    let options = '';
    for(answer in question.answerOptions){
        options +=
        `
            <div class="option">
            <span><b>${answer}</b>: ${question.answerOptions[answer]}</span>
            </div>
        `
    }
    document.querySelector(".question-text").innerHTML = questionTxt; // Değişkenlerde yazdığımız html kodunu atadık.
    document.querySelector(".option-list").innerHTML = options;
    option = optionList.querySelectorAll(".option");
    for(let opt of option) {
        opt.setAttribute("onclick", "optionSelected(this)")
    }
}

const quiz = new Quiz(Questions);
let questionCount = quiz.Questions.length;
correctAnswerCount = 0;
let questionPoint = 100.0 / questionCount;
document.querySelector(".btn-start").addEventListener("click",function(){
    document.querySelector(".score-text").innerHTML = `0 Puan`;
    document.querySelector(".quiz-box").classList.add("active")   
    startTimer(10);
    document.querySelector(".btn-start").classList.add("deactive")
    showQuestion(quiz.getQuestion());
    questionCountShow(quiz.questionIndex+1,quiz.Questions.length);
    document.querySelector(".btn-next-question").classList.add("deactive")
    
});
document.querySelector(".btn-next-question").addEventListener("click",function(){
    if(quiz.Questions.length != (quiz.questionIndex+1)){
        quiz.questionIndex += 1
        startTimer(10);
        showQuestion(quiz.getQuestion());
        questionCountShow(quiz.questionIndex+1,quiz.Questions.length);
        document.querySelector(".btn-next-question").classList.add("deactive")
    }
    else{
        console.log("Quiz Finish.");
        document.querySelector(".quiz-box").classList.add("deactive")
        document.querySelector(".score-box").classList.add("deactive")
    }
}); 

function optionSelected(option){
    clearInterval(counter);
    let answer = option.querySelector("span b").textContent;
    let question = quiz.getQuestion();
    if(question.checkedAnswer(answer)){
        option.classList.add("correct");
        correctAnswerCount += 1;
        document.querySelector(".score-text").innerHTML = `${correctAnswerCount*questionPoint} Puan`;
        option.insertAdjacentHTML("beforeend", correctIcon);
        document.querySelector(".btn-next-question").classList.remove("deactive");
    }
    else{   
        option.classList.add("incorrect");
        option.insertAdjacentHTML("beforeend", incorrectIcon);
        document.querySelector(".btn-next-question").classList.remove("deactive");
    }

    for(let i = 0; i < optionList.children.length;i++){
        optionList.children[i].classList.add("disabled")
    }
}

function questionCountShow(questionNumber, questionCount){
    let tag = `<span class="badge">${questionNumber} / ${questionCount}</span>`;
    document.querySelector(".question-index").innerHTML = tag;
}
time_text = document.querySelector(".time-text")
time_second = document.querySelector(".time-second")

let counter;
function startTimer(time){
    counter = setInterval(timer,1000);
    function timer(){
        time_second.textContent = time;
        time--;
        if(time < 0){
            clearInterval(counter);
            time_text.textContent = "Finish Time"

            let answer = quiz.getQuestion().correctAnswer;
            for(let option of optionList.children){
                if(option.querySelector("span b").textContent == answer){
                    option.classList.add("correct")
                    option.insertAdjacentHTML("beforeend",correctIcon);
                }
                if(option.querySelector("span b").textContent != answer){
                    option.classList.add("incorrect")
                    option.insertAdjacentHTML("beforeend",incorrectIcon);
                }
                option.classList.add("disabled");
                document.querySelector(".btn-next-question").classList.remove("deactive")
            }
        }
    }

}

