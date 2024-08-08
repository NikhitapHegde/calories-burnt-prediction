$(document).ready(function() {
    $("#predictionForm").submit(function(event) {
      event.preventDefault(); // Prevent default form submission
  
      // Get user input data
      const age = parseFloat($("#age").val());
      const gender = $("#gender").val();
      const height = parseFloat($("#height").val());
      const heartrate = parseInt($("#heartrate").val());
      const bodytemp = parseFloat($("#bodytemp").val());
  
      // Send data to Flask app using AJAX
      $.ajax({
        url: "/predict",  // Assuming your Flask app's prediction route is '/predict'
        method: "POST",
        dataType: "json",
        data: {
          age: age,
          gender: gender,
          height: height,
          heartrate: heartrate,
          bodytemp: bodytemp,
        },
        success: function(response) {
          if (response.error) {
            $("#prediction").text("Error: " + response.error);
          } else {
            const predictedCalories = response.predicted_calories;
            $("#prediction").text("Predicted Calories: " + predictedCalories.toFixed(2));
          }
        },
        error: function(jqXHR, textStatus, errorThrown) {
          $("#prediction").text("Error: " + textStatus + " - " + errorThrown);
        }
      });
    });
  });
  