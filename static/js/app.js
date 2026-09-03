/* =========================================================
   ChurnShield — Shared frontend logic
   Flask backend handles real prediction
   ========================================================= */


/* =========================================================
   Chart.js global defaults
   ========================================================= */

function setupCharts() {

    if (!window.Chart) return;

    Chart.defaults.font.family =
        "'Plus Jakarta Sans','Inter',system-ui,sans-serif";

    Chart.defaults.font.weight = "600";
    Chart.defaults.color = "#64748b";

    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.padding = 16;

    Chart.defaults.plugins.tooltip.backgroundColor =
        "rgba(15,23,42,0.92)";

    Chart.defaults.plugins.tooltip.padding = 12;
    Chart.defaults.plugins.tooltip.cornerRadius = 10;
}


/* =========================================================
   Scroll reveal
   ========================================================= */

function initReveal() {

    const elements = document.querySelectorAll(".reveal");

    if (!("IntersectionObserver" in window)) {

        elements.forEach(function(element) {
            element.classList.add("in");
        });

        return;
    }

    const observer = new IntersectionObserver(
        function(entries) {

            entries.forEach(function(entry) {

                if (entry.isIntersecting) {

                    entry.target.classList.add("in");

                    observer.unobserve(entry.target);
                }
            });

        },
        {
            threshold: 0.12
        }
    );

    elements.forEach(function(element) {
        observer.observe(element);
    });
}


/* =========================================================
   Confidence Gauge
   ========================================================= */

window.drawGauge = function(canvas, percent, color) {

    if (!canvas) return;

    const ctx = canvas.getContext("2d");

    const size = 220;

    canvas.width = size;
    canvas.height = size;

    const cx = size / 2;
    const cy = size / 2;
    const radius = 92;

    const start = Math.PI * 0.75;
    const end = Math.PI * 2.25;

    const safePercent = Math.max(
        0,
        Math.min(100, Number(percent) || 0)
    );

    const target =
        start +
        (end - start) *
        (safePercent / 100);

    let current = start;


    function clearGauge() {

        ctx.clearRect(
            0,
            0,
            size,
            size
        );

        ctx.beginPath();

        ctx.lineWidth = 16;
        ctx.strokeStyle = "#eef2ff";
        ctx.lineCap = "round";

        ctx.arc(
            cx,
            cy,
            radius,
            start,
            end
        );

        ctx.stroke();
    }


    function drawFrame() {

        clearGauge();

        const gradient =
            ctx.createLinearGradient(
                0,
                0,
                size,
                size
            );

        gradient.addColorStop(
            0,
            "#4f46e5"
        );

        gradient.addColorStop(
            1,
            color || "#9333ea"
        );

        ctx.beginPath();

        ctx.lineWidth = 16;
        ctx.strokeStyle = gradient;
        ctx.lineCap = "round";

        ctx.arc(
            cx,
            cy,
            radius,
            start,
            Math.min(current, target)
        );

        ctx.stroke();


        if (current < target) {

            current +=
                (target - start) * 0.04;

            requestAnimationFrame(drawFrame);

        } else {

            ctx.beginPath();

            ctx.fillStyle = "#ffffff";

            ctx.arc(
                cx + Math.cos(target) * radius,
                cy + Math.sin(target) * radius,
                8,
                0,
                Math.PI * 2
            );

            ctx.fill();
        }
    }

    drawFrame();
};


/* =========================================================
   Page initialization
   ========================================================= */

document.addEventListener(
    "DOMContentLoaded",
    function() {


        /* =====================================================
           Charts
           ===================================================== */

        setupCharts();


        /* =====================================================
           Scroll animation
           ===================================================== */

        initReveal();


        /* =====================================================
           Navigation
           ===================================================== */

        let currentPage =
            window.location.pathname
                .split("/")
                .pop();

        if (!currentPage) {
            currentPage = "index";
        }

        currentPage =
            currentPage.replace(".html", "");

        document
            .querySelectorAll(
                ".navbar-glass .nav-link"
            )
            .forEach(function(link) {

                const href =
                    link.getAttribute("href");

                if (!href) return;

                const cleanHref =
                    href
                        .replace(".html", "")
                        .replace("./", "")
                        .replace("/", "");

                if (
                    cleanHref === currentPage ||
                    (
                        currentPage === "index" &&
                        cleanHref === ""
                    )
                ) {

                    link.classList.add("active");
                }
            });


        /* =====================================================
           REAL PREDICTION FORM
           ===================================================== */

        const predictForm =
            document.getElementById(
                "predictForm"
            );

        if (predictForm) {

            console.log(
                "Prediction form connected to Flask."
            );

            /*
             IMPORTANT:

             We intentionally DO NOT add:

                 e.preventDefault()

             Flask must receive the form through:

                 POST /predict

             The browser will submit the form normally.
            */
        }


        /* =====================================================
           Tenure Filter
           ===================================================== */

        const tenureFilter =
            document.getElementById(
                "tenureFilter"
            );

        if (tenureFilter) {

            tenureFilter.addEventListener(
                "change",
                function() {

                    const selectedTenure =
                        this.value;

                    const url =
                        new URL(
                            window.location.href
                        );

                    url.searchParams.set(
                        "tenure",
                        selectedTenure
                    );

                    window.location.href =
                        url.toString();
                }
            );
        }


        /* =====================================================
           Analytics Filter Button
           ===================================================== */

        const filterBtn =
            document.getElementById(
                "filterBtn"
            );

        if (filterBtn) {

            filterBtn.addEventListener(
                "click",
                function() {

                    const currentFilter =
                        new URLSearchParams(
                            window.location.search
                        ).get("tenure") || "all";


                    const filter =
                        prompt(

                            "Select tenure filter:\n\n" +

                            "all = All Customers\n" +

                            "0-12 = 0–12 Months\n" +

                            "13-24 = 13–24 Months\n" +

                            "25-36 = 25–36 Months\n" +

                            "37-48 = 37–48 Months\n" +

                            "49-60 = 49–60 Months\n" +

                            "61-72 = 61–72 Months",

                            currentFilter
                        );


                    if (!filter) return;


                    const validFilters = [

                        "all",
                        "0-12",
                        "13-24",
                        "25-36",
                        "37-48",
                        "49-60",
                        "61-72"

                    ];


                    if (
                        !validFilters.includes(
                            filter
                        )
                    ) {

                        alert(
                            "Invalid filter selected."
                        );

                        return;
                    }


                    window.location.href =
                        "/analytics?tenure=" +
                        encodeURIComponent(
                            filter
                        );
                }
            );
        }

    }
);