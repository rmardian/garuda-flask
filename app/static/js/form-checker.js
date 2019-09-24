/**  THIS SCRIPT FOR AUTOMATED CLASSIFICATION/REGRESSION OPTION
//checking when submitting, if there is no target variable selected, or when multiple targets are selected.
$('.run').on('click', function() {
    var checked = $('.targetVar').find(':checked').length;
    if (!checked) {
        alert('No target selected. Please pick your target variable!');
        event.preventDefault();
    }
    else if (checked > 1) {
        //confirmation = confirm(checked + ' variables are checked! Run multi-label task?');
        //if (!confirmation)
         //   event.preventDefault();
        alert('Multi-label is not supported at the moment. Please pick one target variable!');
        event.preventDefault();
    }

    var tune = document.getElementById('hyperpars').checked
    var m = document.getElementById('reqParam0').value
    var a = document.getElementById('reqParam1').value
    var b = document.getElementById('reqParam2').value
    var c = document.getElementById('reqParam3').value
    var d = document.getElementById('reqParam4').value

    if (m == null || m == "") {
        alert("Please fill the model name!");
        event.preventDefault();
    }

    if(tune) {

        if (a == null || a == "" || b == null || b == "" || c == null || c == "" || d == null || d == "") {
            
            alert("Please fill all required fields (NN Parameters Setting)!");
            event.preventDefault();
        }
    }
});
*/

//for now single target variable
function checkTarget() {
    var target = document.getElementById('target_select').value;
    if (target == 'empty') {
        alert('No target selected. Please pick a target variable!');
        event.preventDefault();
    }
}

function checkApproach(val) {

    var checked = $('.targetVar').find(':checked').length;
    var cls = "none";
    var mccls = "none";
    var mlcls = "none";
    var regcls = "none";

    if (checked == 1){
        if (val == 2) {
            cls = "block";
        }
        else if (val > 2) {
            mccls = "block";
        }
        else {
            alert('Value error! selected variable has unacceptable values!')
        }
    }
    else if (checked > 1){
        mlcls = "block";
    }

    document.getElementById("cls_info").style.display = cls;
    document.getElementById("mccls_info").style.display = mccls;
    document.getElementById("mlcls_info").style.display = mlcls;
    document.getElementById("reg_info").style.display = reg;
}

$(document).ready(function() {
    function updateSum() {
        //var total = 0,
        targets = [];
        var i = 0;
        $(".targetVar:checked").each(function(i, n) {
            //total += parseInt($(n).val());
        
            // declare variables for DOM objects and name of crime
            /*var $cb = $(this),
                $row = $cb.parents('tr'),
                name = $row.find('td').last().text();*/
            name = 'test';

            // check that the target-variables array doesn't already have the value; add if necessary
            if (targets.filter(function(c, i, a) {
                    return c == name;
                }).length == 0) {
                    targets.push(name);
                    alert(targets)
                }
        });
        /*
        if (total < 201) {
        $("#time").val(total);
        } else {
        $("#time").val(200);
        }
        $('#charges').val(targets.join(','));*/
    }

    // run the update on every checkbox change and on startup
    $(".targetVar").change(updateSum);
    updateSum();
})


$(document).ready(function() {
    // call onload or in script segment below form

    function attachCheckboxHandlers() {


        
        // get reference to element containing toppings checkboxes
        var el = document.getElementByClassName('targetVar');

        alert(el[0]);
        // get reference to input elements in toppings container element
        var tops = el.getElementsByTagName('input');


        alert(tops.length);

        
        // assign updateTotal function to onclick property of each checkbox
        for (var i=0, len=tops.length; i<len; i++) {
            if ( tops[i].type === 'checkbox' ) {
                tops[i].onclick = updateTotal;
            }
        }
    }
        
    // called onclick of toppings checkboxes
    function updateTotal(e) {

        alert('YES');
        // 'this' is reference to checkbox clicked on
        /*var form = this.form;
        
        // get current value in total text box, using parseFloat since it is a string
        var val = parseFloat( form.elements['total'].value );
        
        // if check box is checked, add its value to val, otherwise subtract it
        if ( this.checked ) {
            val += parseFloat(this.value);
        } else {
            val -= parseFloat(this.value);
        }
        // format val with correct number of decimal places
        // and use it to update value of total text box
        form.elements['total'].value = val;*/
    }
    attachCheckboxHandlers();
})

//Generate metrics
function populateMetrics(){

    var list1 = document.getElementById("mode_select");
    var list2 = document.getElementById("metrics_select");
    var list1SelectedValue = list1.options[list1.selectedIndex].value;

    if (list1SelectedValue=='classification')
    {
        list2.options.length=0;
        list2.options[0] = new Option('Precision', 'precision');
        list2.options[1] = new Option('Recall', 'recall');
        list2.options[2] = new Option('F1 Score', 'f1');
        list2.options[3] = new Option('ROC-AUC', 'roc_auc');
        //list2.options[4] = new Option('MCC', 'mcc');
        //list2.options[5] = new Option('KS Score', 'ks');
    }
    else if (list1SelectedValue=='regression')
    {
        list2.options.length=0;
        list2.options[0] = new Option('MSE', 'neg_mean_squared_error');
        list2.options[1] = new Option('MAE', 'neg_mean_absolute_error');
        //list2.options[2] = new Option('MAPE', 'neg_mean_absolute_error');
        list2.options[2] = new Option('R-squared', 'r2');
    }
}

function activateFold() {
    var val = document.getElementById("cv_select").selectedIndex;
    if (val == 1) {
        document.getElementById("fold_text").disabled = false;
    }
    else {
        document.getElementById("fold_text").disabled = true;
    }
}

function updateModelTable() {
    //var hiddenDiv = document.getElementById("showMe");
    //hiddenDiv.style.display = (this.value == "") ? "none":"block";
    alert(document.getElementById("model_select"));
};

$('#model_select').change(function(){
    var str = $('#model_select').find(":selected").text();
    tables = document.getElementsByClassName('model-table')
    for (i = 0; i < tables.length; i++) {
        document.getElementById(tables[i].id).style.display = "none";
    }
    document.getElementById(str.split(' - ')[0]).style.display = "block";
});
