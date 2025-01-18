$(document).ready(function () {
  // Handle navigation clicks
  $("#homeLink").click(function () {
    $("#homeSection").show();
    $("#uploadSection").show();
    $("#aboutSection").hide();
  });

  $("#aboutLink").click(function () {
    $("#homeSection").hide();
    $("#uploadSection").hide();
    $("#aboutSection").show();
  });

  // Form submission logic
  $("#uploadForm").submit(function (event) {
    event.preventDefault();
    $("#loader").removeClass("hidden");
    $("#predictionResult").empty();

    var formData = new FormData(this);

    $.ajax({
      url: "https://plant-disease-6hpz.onrender.com/predict", // Updated URL to your deployed service
      type: "POST",
      data: formData,
      success: function (response) {
        $("#loader").addClass("hidden");
        $("#predictionResult").html(
          "<h3>Prediction: " + response.prediction + "</h3>"
        );
      },
      error: function () {
        $("#loader").addClass("hidden");
        $("#predictionResult").html(
          "<h3>Error: Unable to analyze the image.</h3>"
        );
      },
      cache: false,
      contentType: false,
      processData: false,
    });
  });
});
